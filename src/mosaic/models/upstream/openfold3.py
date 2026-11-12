"""Featurize OpenFold3 inputs with OpenFold3's own inference pipeline, which needs torch.

The pipeline is jopenfold3's vendored copy of upstream's data code. `features` takes one
query of OpenFold3's inference query JSON as a dict, so it supports everything upstream
does, including nucleic acids, ligands, non-canonical residues, covalent bonds,
precomputed MSAs and templates. As in `run_openfold` without the MSA server, a protein or
RNA chain without `main_msa_file_paths` gets an MSA containing only its query. Template
structures are read from `template_structures` and downloaded from the RCSB if missing.
`features` returns the same features as `OpenFold3.target_only_features`, with a structure
writer.

For design, put the binder first as a chain of `X` residues, as `OpenFold3.binder_features`
does, since `OpenFold3.set_sequence` overwrites the first `PSSM.shape[0]` tokens. For
example, to design a binder to a protein with an MSA and bound ATP:

    import jax

    from mosaic.models.openfold3 import OpenFold3
    from mosaic.models.upstream import openfold3 as upstream

    model = OpenFold3()
    features, writer = upstream.features(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": "A", "sequence": "X" * 80},
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": target_sequence,
                    "main_msa_file_paths": "msas/target/colabfold_main.a3m",
                },
                {"molecule_type": "ligand", "chain_ids": "C", "ccd_codes": "ATP"},
            ]
        },
        key=jax.random.key(0),
    )
    loss = model.build_loss(loss=..., features=features)
    prediction = model.predict(PSSM=pssm, features=features, writer=writer, key=key)

Upstream only reads MSA files with names it recognizes, such as `colabfold_main` or
`uniref90_hits`."""

import tempfile
import warnings
from pathlib import Path

import gemmi
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import PRNGKeyArray
from jfold.featurize import AtomSites
from jopenfold3._vendor.openfold3.core.data.framework.single_datasets.inference import (
    InferenceDataset,
)
from jopenfold3._vendor.openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from jopenfold3._vendor.openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from jopenfold3._vendor.openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from jopenfold3._vendor.openfold3.core.data.tools.colabfold_msa_server import (
    MsaComputationSettings,
    augment_main_msa_with_query_sequence,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
    TemplateSettings,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceJobConfig,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)

from mosaic.cache import cache_dir
from mosaic.models.jfold_featurize import AtomSitesWriter
from mosaic.models.openfold3 import backend_features
from mosaic.models.upstream import global_rngs


class _InferenceDataset(InferenceDataset):
    """Upstream's inference dataset, fixed to read templates from the preprocessor's cache.

    The vendored copy does not pass the cache directory, which drops every template.
    Upstream OpenFold3 passes it."""

    def create_template_features(self, query, atom_array, n_tokens):
        settings = self.template_settings
        preprocessed = self.template_preprocessor_settings
        assembly_data = {
            chain_id: {
                "template_ids": chain.template_entry_chain_ids,
                "cache_entry_file_path": chain.template_alignment_file_path,
            }
            for chain in query.chains
            for chain_id in chain.chain_ids
        }
        templates = process_template_structures_of3(
            atom_array=atom_array,
            n_templates=settings.n_templates,
            take_top_k=settings.take_top_k,
            min_n_tokens_per_chain=settings.min_n_tokens_per_chain,
            template_cache_directory=preprocessed.cache_directory,
            assembly_data=assembly_data,
            template_structures_directory=preprocessed.structure_directory,
            template_structure_array_directory=preprocessed.structure_array_directory,
            template_file_format=preprocessed.structure_file_format,
            ccd=self.ccd,
        )
        return featurize_template_structures_of3(
            atom_array=atom_array,
            template_slice_collection=templates,
            n_templates=settings.n_templates,
            n_tokens=n_tokens,
            min_bin=settings.distogram.min_bin,
            max_bin=settings.distogram.max_bin,
            n_bins=settings.distogram.n_bins,
        )


def preprocess_templates(
    queries: InferenceQuerySet, settings: TemplatePreprocessorSettings
) -> None:
    """Run upstream's template preprocessing for the template hits in `queries`.

    This runs in the current process, one chain at a time, because `TemplatePreprocessor()`
    forks a process pool, which is unsafe once JAX has started threads. It also raises if
    a chain's templates are dropped, where upstream only prints a message."""
    chains = queries.queries["query"].chains
    templated = [c for c in chains if c.template_alignment_file_path is not None]
    if not templated:
        return
    preprocessor = TemplatePreprocessor(input_set=queries, config=settings)
    for entry in preprocessor.inputs:
        preprocessor.preprocess_templates(entry)
    preprocessor._update_inference_query_set()
    for chain in templated:
        if not chain.template_entry_chain_ids:
            raise RuntimeError(
                f"OpenFold3's template preprocessing dropped chain {chain.chain_ids[0]}'s "
                "templates"
            )


def atom_sites(atom_array: AtomArray) -> AtomSites:
    """Convert upstream's atom array to `AtomSites`, keeping the feature order."""
    return AtomSites(
        chain_id=tuple(map(str, atom_array.chain_id)),
        residue_name=tuple(map(str, atom_array.res_name)),
        residue_number=np.asarray(atom_array.res_id, np.int64),
        atom_name=tuple(map(str, atom_array.atom_name)),
        element=np.array([gemmi.Element(e).atomic_number for e in atom_array.element]),
        token=np.asarray(atom_array.token_id, np.int64),
    )


def features(
    input: dict, *, key: PRNGKeyArray, template_structures: Path | None = None
) -> tuple[dict, AtomSitesWriter]:
    """Featurize the OpenFold3 query `input` and return the features and a writer.

    `key` seeds the RNGs that generate and randomly place the reference conformers. The
    global RNGs are restored afterwards. `template_structures` is the directory that
    template structures are read from and downloaded to. If None, a directory in mosaic's
    cache is used."""
    if template_structures is None:
        template_structures = cache_dir("openfold3", "template_structures")
    queries = InferenceQuerySet(queries={"query": Query.model_validate(input)})
    with (
        tempfile.TemporaryDirectory() as tmp,
        global_rngs(key, seed_torch=True) as seed,
    ):
        directory = Path(tmp)
        with warnings.catch_warnings():
            # It warns for each chain that gets a query-only MSA, such as a binder.
            warnings.simplefilter("ignore")
            queries = augment_main_msa_with_query_sequence(
                queries, MsaComputationSettings(msa_output_directory=directory / "msas")
            )
        templates = TemplatePreprocessorSettings(
            output_directory=directory / "template_data",
            structure_directory=template_structures,
        )
        preprocess_templates(queries, templates)
        dataset = _InferenceDataset(
            InferenceJobConfig(
                query_set=queries,
                seeds=[seed],
                msa=MSASettings(subsample_main=False),
                template=TemplateSettings(take_top_k=True),
                template_preprocessor_settings=templates,
            )
        )
        batch = dataset.create_all_features(queries.queries["query"])
    atom_array = batch.pop("atom_array")
    return (
        backend_features({k: v.numpy() for k, v in batch.items()}),
        AtomSitesWriter(atom_sites(atom_array)),
    )
