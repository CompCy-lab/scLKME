scLKME: Landmark-based Multi-sample Single-cell Data Analysis
=============================================================

**scLKME** is an approach for sample-level analysis of multi-sample single-cell data. It uses
landmark-based kernel mean embedding to generate sample embeddings. scLKME includes two steps:

1. `cell sketching`: identify a subset of cells as landmarks to summarize the cell landscape across samples.
2. `kernel mean embedding`: transform cell distributions using kernel mean embedding and align them at the landmarks.

The workflow of **scLKME** is as follows:

.. image:: https://raw.githubusercontent.com/CompCy-lab/scLKME/main/docs/_static/img/scLKME_workflow.png
   :alt: scLKME workflow figure
   :width: 800px
   :align: center

~~~~~

Get started
-----------
- To install the :mod:`sclkme`, see the :doc:`installation`.
- For api usage, see the :doc:`api`.
- For data analysis, check out the :doc:`notebooks/tutorials/index`. 


.. toctree::
   :maxdepth: 2
   :caption: Content:
   :hidden:

   installation
   api
   references

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :hidden:

   notebooks/tutorials/index

