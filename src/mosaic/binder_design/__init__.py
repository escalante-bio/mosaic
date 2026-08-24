"""The binder design pipeline, running natively inside Mosaic.

Four stages, all in one Python 3.12 process:

==========  ========================  =============================
stage       module                    what it does
==========  ========================  =============================
1           :mod:`.trajectory`        AF2 hallucination of a binder backbone
2           :mod:`.mpnn`              ProteinMPNN sequence design
3           :mod:`.validation`        refold designs, score confidence
4           :mod:`.rosetta`,          PyRosetta interface scoring, DSSP
            :mod:`.geometry`          secondary structure, clashes, RMSD
==========  ========================  =============================

:mod:`.pipeline` drives stages 2-4 over trajectory backbones, :mod:`.filters`
applies the threshold configs and :mod:`.labels` defines the CSV schema, both
kept compatible with DdCraft so existing configs and analysis scripts still
work.  Run it with ``python -m mosaic.binder_design``.

Because stage 3 only needs a Mosaic ``StructurePredictionModel``, the folding
model is swappable: AlphaFold2 today, Boltz / ESMFold / Protenix / OpenFold3
without touching the rest of the pipeline.

Submodules are deliberately not imported eagerly -- importing PyRosetta and the
AF2 parameters is slow, and most callers only need one stage.
"""
