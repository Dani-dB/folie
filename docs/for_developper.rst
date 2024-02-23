#######################################################
For developper
#######################################################


Run
    >>> git clone git@github.com:langevinmodel/folie.git
    >>> cd folie
    >>> pip install -e .

to install the package for local developpement.


Input of trajectories data
==============================



Building an estimator of Langevin dynamics
============================================


Likelihood estimation: using Transition density
--------------------------------------------------


.. inheritance-diagram:: folie.EulerDensity folie.OzakiDensity folie.ShojiOzakiDensity folie.ElerianDensity folie.KesslerDensity folie.DrozdovDensity
    :top-classes: folie.estimation.transitionDensity.TransitionDensity



Writing a new estimator
-----------------------------------

Estimator should generically inherit from :class`Estimator` and implement a fit method. Since most estimator of folie are flexible they generally take as argument the model to be estimated, but if your estimator is model specific you are not force to follow this approach.

Model of Langevin Dynamics
=============================

Folie holds a number of models for Langevin Dynamics. The main models simply contain reference to functions that describe the spatial dependences as well as the associated coefficients.

There is also some


Writing a new model
-----------------------------------




Writing a new function
---------------------------------
Functions are the core part of folie for the description of spatial dependences. 
Functions can be parametric functions (i.e. with coefficients to be optimized) or non-parametric.


Documenting the code
=============================

When documenting your code, `numpydoc style <numpydoc.readthedocs.io>`__ should be used. Going back to the example
of the :code:`MeanEstimator`, this style of documentation would look like the following:

.. code-block:: python

    class SimpleEstimator(folie.base.Estimator):
        r""" A simple estimator. It estimates the mean using a complicated algorithm
        :footcite:`author1991`.

        Parameters
        ----------
        axis : int, optional, default=-1
            The axis over which to compute the mean. Defaults to -1, which refers to the last axis.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        Overdamped
        """

        def __init__(self, axis=-1):
            super().__init__()
            self.axis = axis

        def fit(self, data):
            r""" Performs the estimation.

            Parameters
            ----------
            data : ndarray
                Array over which the mean should be estimated.

            Returns
            -------
            self : MeanEstimator
                Reference to self.
            """
            self._model = MeanModel(np.mean(data, axis=self.axis))
            return self

Note the specific style of using citations. For citations there is a package-global BibTeX file under
:code:`docs/references.bib`. These references can then be included into the documentation website
using the citation key as defined in the references file.

The documentation website is hosted via GitHub pages. Please see the
`README <https://github.com/langevinmodel/folie/blob/main/README.md>`__ on GitHub for instructions on how to build
it.