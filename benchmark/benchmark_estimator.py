import pytest
import os
import numpy as np
import folie as fl
import dask.array as da
import multiprocessing


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:, i : (i + 1)])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_direct_optimization(data, request, benchmark):
    bf = fl.functions.Linear()
    model = fl.models.Overdamped(bf)
    estimator = fl.KramersMoyalEstimator(model)
    fitted_estimator = benchmark(estimator.fit, data, coefficients0=[1.0, 1.0])
    model = fitted_estimator.fetch_model()
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_likelihood_optimization(data, request, benchmark):
    bf = fl.functions.Linear()
    model = fl.models.Overdamped(bf)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
    fitted_estimator = benchmark(estimator.fit, data, coefficients0=[1.0, 1.0])
    model = fitted_estimator.fetch_model()
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize("n_jobs", [1, 2, 4, -1])
def test_likelihood_optimization_parallel(data, request, n_jobs, benchmark):
    bf = fl.functions.Linear()
    model = fl.models.Overdamped(bf)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model), n_jobs=n_jobs)
    fitted_estimator = benchmark(estimator.fit, data, coefficients0=[1.0, 1.0])
    model = fitted_estimator.fetch_model()
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_em_estimator(data, request, benchmark):
    fun_lin = fl.functions.Linear().fit(data)
    fun_cst = fl.functions.Constant().fit(data).resize((3, 3))
    model = fl.models.OverdampedHidden(fun_lin, fun_lin.copy(), fun_cst, dim=1, dim_h=2)
    estimator = fl.EMEstimator(fl.EulerDensity(model), max_iter=2)
    fitted_estimator = benchmark(estimator.fit, data)
    model = fitted_estimator.fetch_model()
    assert model.fitted_
