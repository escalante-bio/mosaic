"""OpenFold3 featurized by its own inference pipeline (torch; jopenfold3's vendored copy of
upstream's data code) from OpenFold3's own input: one query of its inference query JSON,
as a dict (validated by upstream's `Query`), so whatever upstream takes: proteins, DNA, RNA,
ligands (CCD codes, SMILES, SDF files), non-canonical residues, covalent bonds, precomputed
MSAs (`main_msa_file_paths`, `paired_msa_file_paths`) and templates
(`template_alignment_file_path`). As `run_openfold` without the MSA server, a protein or RNA
chain without `main_msa_file_paths` gets a query-only MSA. Template hits name PDB entries,
read from `template_structures` as `<entry id>.cif` and fetched from the RCSB into it when
missing. The features are those of `mosaic.models.openfold3.OpenFold3.target_only_features`
(upstream's dict, unbatched, with `rep_atom`); the writer writes every entity, from the
atoms of upstream's atom array.

To design, put the binder first as a placeholder chain of `X` residues, as
`OpenFold3.binder_features` does: `OpenFold3.set_sequence` overwrites the first
`PSSM.shape[0]` tokens. For a binder to a protein (with its MSA) with ATP bound:

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

Upstream reads only the MSA files whose stem it knows (e.g. `colabfold_main`,
`uniref90_hits`)."""

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
    """Upstream's inference dataset, reading templates from the cache the template
    preprocessor wrote: the vendored copy passes none, which drops every template (upstream
    OpenFold3 passes it)."""

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
    """Upstream's template preprocessing of `queries`' template hits into its template
    cache, recorded on their chains. Run in this process, one chain after another, where
    `TemplatePreprocessor()` forks a pool (unsafe once JAX runs threads) and only prints a
    template it drops."""
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
    """The atoms of upstream's atom array, in its (the features') order."""
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
    """OpenFold3's features for the query `input` (see the module docstring) and a writer
    of every entity. `key` seeds the reference conformers (RDKit's embedding, from Python's
    RNG) and their random poses (torch's RNG); the global RNGs are restored after.
    `template_structures`: the directory template hits' structures are read from and
    fetched to (default: mosaic's cache, `openfold3/template_structures`)."""
    if template_structures is None:
        template_structures = cache_dir("openfold3", "template_structures")
    queries = InferenceQuerySet(queries={"query": Query.model_validate(input)})
    with (
        tempfile.TemporaryDirectory() as tmp,
        global_rngs(key, seed_torch=True) as seed,
    ):
        directory = Path(tmp)
        with warnings.catch_warnings():
            # It warns of each chain given the query-only MSA, as a design's binder is.
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
