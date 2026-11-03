"""The p-value-checked capacity ratio is gone, with no deprecation cycle.

This is the one test module allowed to name the removed API.
"""

import inspect

import captest.capdata as pvc
from captest import CapTest
from captest.captest import CapTestResults


def test_the_p_check_api_is_gone():
    assert "check_pvalues" not in inspect.signature(CapTest.run_test).parameters
    assert "pval" not in inspect.signature(CapTest.run_test).parameters
    assert "check_pvalues" not in inspect.signature(CapTest.captest_results).parameters
    assert "pval" not in inspect.signature(CapTest.captest_results).parameters
    assert not hasattr(CapTest, "captest_results_check_pvalues")
    assert not hasattr(pvc, "predict_with_pvalue_check")
    assert "cap_ratio_pval_check" not in CapTestResults.__dataclass_fields__
    assert "pvalues_checked" not in CapTestResults.__dataclass_fields__


def test_the_p_value_report_stays():
    assert hasattr(CapTestResults, "styled_pvalues")
