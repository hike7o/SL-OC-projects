# 1. Library imports
import pandas as pd 
from pydantic import BaseModel, Field
from typing import List


# 2. Class which describes a single flower measurements
class Parameters(BaseModel):

        NAME_CONTRACT_TYPE                              : float
        FLAG_OWN_CAR                                    : float
        AMT_INCOME_TOTAL                                : float
        AMT_CREDIT                                      : float
        NAME_TYPE_SUITE                                 : float
        NAME_INCOME_TYPE                                : float
        NAME_FAMILY_STATUS                              : float
        NAME_HOUSING_TYPE                               : float
        REGION_POPULATION_RELATIVE                      : float
        DAYS_BIRTH                                      : float
        DAYS_EMPLOYED                                   : float
        OCCUPATION_TYPE                                 : float
        LIVE_CITY_NOT_WORK_CITY                         : float
        ORGANIZATION_TYPE                               : float
        EXT_SOURCE_1                                    : float
        FLOORSMAX_MODE                                  : float
        TOTALAREA_MODE                                  : float
        AMT_REQ_CREDIT_BUREAU_QRT                       : float
        NEW_DOC_KURT                                    : float
        EXT_SOURCES_PROD                                : float
        EXT_SOURCES_MIN                                 : float
        EXT_SOURCES_MAX                                 : float
        EXT_SOURCES_NANMEDIAN                           : float
        EXT_SOURCES_VAR                                 : float
        CREDIT_TO_ANNUITY_RATIO                         : float
        ANNUITY_TO_INCOME_RATIO                         : float
        INCOME_TO_EMPLOYED_RATIO                        : float
        INCOME_TO_BIRTH_RATIO                           : float
        ID_TO_BIRTH_RATIO                               : float
        CAR_TO_BIRTH_RATIO                              : float
        PHONE_TO_BIRTH_RATIO                            : float
        GROUP_EXT_SOURCES_MEDIAN                        : float
        GROUP_INCOME_MEAN                               : float
        BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM               : float
        BUREAU_LL_DEBT_CREDIT_DIFF_MEAN                 : float
        BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MIN           : float
        BUREAU_ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN       : float
        BUREAU_ACTIVE_DAYS_CREDIT_UPDATE_MIN            : float
        BUREAU_ACTIVE_DAYS_CREDIT_UPDATE_MEAN           : float
        BUREAU_CLOSED_AMT_CREDIT_SUM_DEBT_MAX           : float
        BUREAU_CONSUMER_DEBT_PERCENTAGE_MEAN            : float
        BUREAU_CONSUMER_DAYS_CREDIT_ENDDATE_MAX         : float
        BUREAU_CREDIT_AMT_CREDIT_SUM_MAX                : float
        BUREAU_CREDIT_AMT_CREDIT_SUM_DEBT_MEAN          : float
        BUREAU_CREDIT_AMT_CREDIT_SUM_DEBT_MAX           : float
        BUREAU_LAST6M_AMT_CREDIT_SUM_DEBT_MEAN          : float
        BUREAU_LAST12M_AMT_CREDIT_MAX_OVERDUE_MEAN      : float
        BUREAU_LAST12M_AMT_CREDIT_SUM_DEBT_MEAN         : float
        BUREAU_LAST12M_AMT_CREDIT_SUM_DEBT_SUM          : float
        PREV_DAYS_DECISION_MEAN                         : float
        PREV_CNT_PAYMENT_MAX                            : float
        PREV_APPLICATION_CREDIT_RATIO_MIN               : float
        PREV_NAME_CONTRACT_STATUS_Approved_MEAN         : float
        PREV_ACTIVE_CNT_PAYMENT_MEAN                    : float
        PREV_ACTIVE_DAYS_LAST_DUE_1ST_VERSION_MEAN      : float
        PREV_ACTIVE_INSTALMENT_PAYMENT_DIFF_MEAN        : float
        PREV_ACTIVE_REMAINING_DEBT_MAX                  : float
        APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN           : float
        APPROVED_DAYS_LAST_DUE_DIFF_MAX                 : float
        PREV_Consumer_AMT_CREDIT_SUM                    : float
        PREV_Cash_AMT_ANNUITY_MEAN                      : float
        PREV_Cash_AMT_ANNUITY_MAX                       : float
        PREV_Cash_SIMPLE_INTERESTS_MIN                  : float
        PREV_Cash_APPLICATION_CREDIT_RATIO_MEAN         : float
        PREV_Cash_DAYS_LAST_DUE_1ST_VERSION_MAX         : float
        PREV_LAST24M_AMT_ANNUITY_MAX                    : float
        PREV_LAST24M_DAYS_LAST_DUE_1ST_VERSION_MIN      : float
        PREV_LAST24M_DAYS_LAST_DUE_1ST_VERSION_MEAN     : float
        POS_SK_DPD_VAR                                  : float
        CC_AMT_TOTAL_RECEIVABLE_MEAN                    : float
        CC_CNT_DRAWINGS_ATM_CURRENT_SUM                 : float
        CC_SK_DPD_DEF_MAX                               : float
        CC_LATE_PAYMENT_SUM                             : float
        INS_SK_ID_PREV_SIZE                             : float
        INS_SK_ID_PREV_NUNIQUE                          : float
        INS_AMT_PAYMENT_MAX                             : float
        INS_DBD_MEAN                                    : float
        INS_LATE_PAYMENT_SUM                            : float
        INS_36M_DAYS_ENTRY_PAYMENT_MIN                  : float
        INS_36M_AMT_INSTALMENT_MEAN                     : float
        INS_36M_AMT_PAYMENT_SUM                         : float
        INS_36M_SIGNIFICANT_LATE_PAYMENT_MEAN           : float
        INS_60M_LATE_PAYMENT_MEAN                       : float
        INS_60M_SIGNIFICANT_LATE_PAYMENT_MEAN           : float
        BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO            : float
        CURRENT_TO_APPROVED_CREDIT_MIN_RATIO            : float

class Predproba(BaseModel):
    prediction: int
    probability: float
        
class FeatureImportance(BaseModel):
    zero: List[List[float]] = Field(alias='0')
    one: List[List[float]] =  Field(alias='1')  
    
    
    
    
    
    
    
    
    