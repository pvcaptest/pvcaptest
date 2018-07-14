- bounds check and warnings options of the CapData.__series_type cause plot method to break up sensor types into separate plots if one (some) sensors of the sensor type have value warnings and others don't
- setting of translation dictionary needs refinement
- translation dictionary does not give consistent results
- pvsyst output file- need to separate tarray from inv losses
- need to split pvsyst GlobHor and GlobInc; GlobInc is ending up in irr-ghi-
- Add uncertainty calculation to the summary function
- edit documentation for added features of the plot method
- add standard test that tries to run everything with no user input
- change cp_results to use the CapTest.tolerance
- update cptest.scatter method to make multiple plots if there are multiple poa sensors
- edit documentation of the filter power factor method, not clear exactly what pf data it is filtering on change so user specs trans key to filter on rather than finding any pf in trans_dict
- plotting function to scatter plot regression variables from sim and das together
    - ex: irr vs power for both sim and das on same plot or same for wind, temp
- clean up the filter arguments formatting in the summary method output and remove the 'das' or 'sim' from the beginning
- add an original line to top of summary output for sim and das to show number of data points before first filtering step and change "Timestamps to "Timestamps Before Filter" and "Timestamps_filtered" to "Timestamps Removed"
- change scatter to plot each column as separate series if there is more than one column

`power = das.rview('power')
poa1 = das.rview('poa').iloc[:,0]
poa2 = das.rview('poa').iloc[:,1]

plt.plot(poa1, power, 'ro', alpha=0.2, markersize=3)
plt.plot(poa2, power, 'bo', alpha=0.2, markersize=3)`