
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_ghst.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_ghst.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_ghst.py:


Demo: GHST Distribution
-----------------------

.. GENERATED FROM PYTHON SOURCE LINES 5-17



.. image-sg:: /auto_examples/images/sphx_glr_plot_ghst_001.png
   :alt: plot ghst
   :srcset: /auto_examples/images/sphx_glr_plot_ghst_001.png
   :class: sphx-glr-single-img





.. code-block:: default

    from plotnine import ggplot, aes, geom_area
    from tspydistributions import pdqr

    def ghst_left(x):
        return pdqr.dsghst(x, mu=2, sigma=1, skew=-20, shape=10)[0]

    def ghst_right(x):
        return pdqr.dsghst(x, mu=-2, sigma=1, skew=20, shape=10)[0]

    plot = (ggplot(None,aes([-6,6])) + 
            geom_area(stat = "function", fun = ghst_left, fill = "cadetblue", alpha = 0.4, xlim = [-6, 6]) + 
            geom_area(stat = "function", fun = ghst_right, fill = "darkgrey", alpha = 0.4, xlim = [-6, 6]))
    print(plot)

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 1.686 seconds)


.. _sphx_glr_download_auto_examples_plot_ghst.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example




    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_ghst.py <plot_ghst.py>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_ghst.ipynb <plot_ghst.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
