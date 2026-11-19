"""Every flag name the package writes, defined once (leaf module).

A flag is a boolean column of the sample or protein QC summary, or a counter of the
manifest build report. Modules import these names instead of spelling the strings, so a
typo fails at import. The FAIL and WARN prefixes are what ``save_qc_results`` and the
report use to tell the two apart.
"""

# ── Manifest and clinical exclusion flags ──
FAIL_BRIDGE = "FAIL_Bridge_Sample"
FAIL_NO_MANIFEST_MATCH = "FAIL_No_Manifest_Match"
# Table-only: a tube with no assay data has no sample-summary row to flag.
FAIL_NOT_ASSAYED = "FAIL_Not_Assayed"
FAIL_NO_TREATMENT_ARM = "FAIL_No_Treatment_Arm"
FAIL_NO_ADSL_MATCH = "FAIL_No_ADSL_Match"
FAIL_NO_TREATMENT_WINDOW = "FAIL_No_Treatment_Window"
FAIL_NO_ADT = "FAIL_No_ADT"
FAIL_AFTER_TREATMENT = "FAIL_After_Treatment"
FAIL_NOT_ANALYSIS_VISIT = "FAIL_Not_Analysis_Visit"
FAIL_DUPLICATE_VISIT = "FAIL_Duplicate_Visit"
FAIL_BASELINE_AFTER_DOSE = "FAIL_Baseline_After_Dose"
FAIL_NO_BASELINE = "FAIL_No_Baseline"
FAIL_ONLY_BASELINE = "FAIL_Only_Baseline"
# An ``exclude`` correction outranks every flag and the exclusion table labels it with
# the correction's ``label``, so this flag is not in the table's reason list.
CORRECTION_EXCLUDE_FLAG = "FAIL_Correction_Exclude"

# ── Identity flags ──
FAIL_IDENTITY_FOREIGN = "FAIL_Identity_Foreign"
FAIL_IDENTITY_UNRESOLVED = "FAIL_Identity_Unresolved"
WARN_IDENTITY_VETOED = "WARN_Identity_Vetoed"
WARN_IDENTITY_AMBIGUOUS = "WARN_Identity_Ambiguous"
WARN_IDENTITY_PQTL = "WARN_Identity_pQTL"

# ── Technical sample flags ──
FAIL_SAMPLE_QC = "FAIL_Sample_QC"
FAIL_PCA_OUTLIER = "FAIL_PCA_Outlier"
FAIL_MEDIAN_IQR_OUTLIER = "FAIL_Median_IQR_Outlier"
FAIL_ROWCHECK = "FAIL_RowCheck"
WARN_SAMPLE_QC = "WARN_Sample_QC"
WARN_MISSING_RATE = "WARN_Missing_Rate"
WARN_LOW_VOLUME = "WARN_Low_Volume"
WARN_RFU_OUTLIER = "WARN_RFU_Outlier"

# ── Concordance flags ──
WARN_SEX_CONCORDANCE = "WARN_Sex_Concordance"
WARN_AGE_CONCORDANCE = "WARN_Age_Concordance"
WARN_AGE_PREDICTION = "WARN_Age_Prediction"
WARN_AGE_CONSISTENCY = "WARN_Age_Consistency"

# ── Protein (assay) flags ──
FAIL_ASSAY_QC = "FAIL_Assay_QC"
WARN_ASSAY_QC = "WARN_Assay_QC"
FAIL_DUPLICATE_ASSAY = "FAIL_Duplicate_Assay"
FAIL_LOW_DETECTION = "FAIL_Low_Detection"
FAIL_INTRA_CV_AND_DETECTION_RATE = "FAIL_Intra_CV_and_Detection_Rate"
WARN_DETECTION_RATE = "WARN_Detection_Rate"
WARN_INTRA_CV = "WARN_Intra_CV"
WARN_COLCHECK = "WARN_ColCheck"
# SomaScan ``finalize_assay_flags`` writes the per-check flags as ``WARN_<check>``.
FAIL_MULTIPLE_FLAGS = "FAIL_Multiple_Flags"
WARN_SATURATION = "WARN_Saturation"
WARN_SB_RATIO = "WARN_SB_Ratio"

# ── Counters of the manifest build report (manifest.py); not summary columns ──
WARN_ADLB_VISIT_MULTIPLE_DATES = "WARN_ADLB_Visit_Multiple_Dates"
WARN_ADT_FILL_ADLB_VISIT = "WARN_ADT_Fill_ADLB_Visit"
WARN_ADT_FILL_ADVS = "WARN_ADT_Fill_ADVS"
WARN_ADVS_VISIT_MULTIPLE_DATES = "WARN_ADVS_Visit_Multiple_Dates"
WARN_DATE_MATCH_NO_HIT = "WARN_Date_Match_No_Hit"
WARN_DATE_MATCH_UNSCHEDULED_ONLY = "WARN_Date_Match_Unscheduled_Only"
WARN_DUPLICATE_SUBJECT_VISIT = "WARN_Duplicate_Subject_Visit"
WARN_PLANNED_VISIT_DIFFERS = "WARN_Planned_Visit_Differs"
WARN_SITE_TRANSFER = "WARN_Site_Transfer"
WARN_TUBE_ADLB_VISIT_DATE_DISAGREE = "WARN_Tube_ADLB_Visit_Date_Disagree"
WARN_TUBE_RECORDS_DISAGREE = "WARN_Tube_Records_Disagree"
WARN_VENDOR_DATE_UNPARSED = "WARN_Vendor_Date_Unparsed"
WARN_VENDOR_TUBE_DATE_DISAGREE = "WARN_Vendor_Tube_Date_Disagree"
WARN_VISIT_DATE_ORDER = "WARN_Visit_Date_Order"
WARN_VISIT_LABEL_UNPARSED = "WARN_Visit_Label_Unparsed"
FAIL_NO_ADSL_SUBJECT = "FAIL_No_ADSL_Subject"
FAIL_NULL_ADT = "FAIL_Null_ADT"
FAIL_TWO_SCHEDULED_VISITS_SAME_DATE = "FAIL_Two_Scheduled_Visits_Same_Date"

# The manifest build report's counters, which are not columns of the QC summaries.
MANIFEST_REPORT_FLAGS = (
    WARN_ADLB_VISIT_MULTIPLE_DATES,
    WARN_ADT_FILL_ADLB_VISIT,
    WARN_ADT_FILL_ADVS,
    WARN_ADVS_VISIT_MULTIPLE_DATES,
    WARN_DATE_MATCH_NO_HIT,
    WARN_DATE_MATCH_UNSCHEDULED_ONLY,
    WARN_DUPLICATE_SUBJECT_VISIT,
    WARN_PLANNED_VISIT_DIFFERS,
    WARN_SITE_TRANSFER,
    WARN_TUBE_ADLB_VISIT_DATE_DISAGREE,
    WARN_TUBE_RECORDS_DISAGREE,
    WARN_VENDOR_DATE_UNPARSED,
    WARN_VENDOR_TUBE_DATE_DISAGREE,
    WARN_VISIT_DATE_ORDER,
    WARN_VISIT_LABEL_UNPARSED,
    FAIL_NO_ADSL_SUBJECT,
    FAIL_NULL_ADT,
    FAIL_TWO_SCHEDULED_VISITS_SAME_DATE,
)
