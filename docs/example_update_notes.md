complete_capacity_test.ipynb
-   Replace the call to set_regression_cols and subsequent agg_sensors to get aggregated regression columns with setting regression columns to hand written dict that contains aggregation tuples and then call process regression_cols. THIS WAS CHANGED IN commit 4590311935238dd84d706c581a0f5fe610af7ef1 (unreleased first commit after v0.14).
- small cleanup improvement to filter_irr - the ref_val kwarg now accepts the more clear string "rep_irr" and the actual reporting irradiance value is included in the summary when running get_summary method
- Functions comparing across two instances of CapData have been moved to CapTest methods - e.g. `capdata.get_summary` and `capdata.captest_results_check_pvalues`
- Calculating reporting conditions for multiple time periods (e.g., monthly) moved out of `rep_cond` to `rep_cond_freq`
- 
