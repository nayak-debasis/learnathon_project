import pandas as pd
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay # Added ConfusionMatrixDisplay
import matplotlib.pyplot as plt # Added matplotlib for visualizations

# Re-define the content of the files as the previous script's output is not directly accessible
file01_content = """Claim_ID,Bind_Date1,Customer_Life_Value1,Age_Insured,Policy_Num,Policy_State,Policy_Start_Date,Policy_Expiry_Date,Policy_BI,Policy_Ded,Policy_Premium,Umbrella_Limit,Insured_Zip,Gender,Education,Occupation,Hobbies,Insured_Relationship,Capital_Gains,Capital_Loss,Garage_Location,Accident_Date,Accident_Type,Collision_Type,Accident_Severity,authorities_contacted,Acccident_State,Acccident_City,Accident_Location,Accident_Hour,Num_of_Vehicles_Involved,Property_Damage,Bodily_Injuries,Witnesses,Police_Report,DL_Expiry_Date,Claims_Date,Auto_Make,Auto_Model,Auto_Year,Vehicle_Color,Vehicle_Cost,Annual_Mileage,DiffIN_Mileage,Low_Mileage_Discount,Fraud_Ind,Commute_Discount,Total_Claim,Injury_Claim,Property_Claim,Vehicle_Claim,Vehicle_Registration,Check_Point
AA00000001,1/1/2023,12,28,123790687,OH,10/13/2023,4/13/2024,500/1000,1000,986.53,0,472720,FEMALE,High School,adm-clerical,polo,other-relative,62700,0,No,2/16/2024,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,VA,Columbus,7819 Oak St,11,1,NO,0,2,YES,8/12/2025,2/17/2024,Honda,CRV,2016,White,6368.88,15093,2499,0,N,0,6162.56,714.94,5370.74,76.88,IF47V1395,No
AA00000002,1/1/2023,12,31,129044473,IL,10/21/2023,4/21/2024,250/500,500,1163.83,0,604874,MALE,Associate,protective-serv,movies,husband,37700,0,No,2/21/2024,Multi-vehicle Collision,Side Collision,Major Damage,Other,NC,Arlington,7609 Rock St,21,4,YES,2,0,,4/15/2026,2/26/2024,Audi,A5,2021,Black,14285.98,5824,4242,1,N,0,20402.38,7669.31,5708.22,7024.85,EI51L7783,No
AA00000003,7/1/2022,18,50,146863149,OH,11/26/2023,5/26/2024,500/1000,500,889.13,0,459429,FEMALE,Masters,priv-house-serv,board-games,other-relative,0,0,No,2/26/2024,Multi-vehicle Collision,Rear Collision,Total Loss,Police,PA,Northbend,4618 Flute Ave,14,3,NO,0,2,NO,4/24/2026,3/1/2024,Audi,A3,2024,Gray,24293.57,10527,2346,0,N,0,10839.12,3646.11,3468.94,3724.07,MU37B8905,No
AA00000004,1/1/2023,12,37,163100869,IL,8/8/2023,2/8/2024,500/1000,1000,1060.74,0,471585,MALE,PhD,tech-support,reading,own-child,0,-51500,No,1/9/2024,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Columbus,1229 5th Ave,15,1,YES,2,3,,3/17/2026,1/13/2024,Suburu,Impreza,2024,Gray,24526.36,11603,2425,0,Y,0,17423.88,5585.62,1863.46,9974.8,RI52Q2108,No
AA00000005,3/1/2022,22,28,185582958,OH,11/12/2023,5/12/2024,250/500,1000,1235.14,0,443567,MALE,MD,exec-managerial,camping,husband,0,-32100,No,2/17/2024,Multi-vehicle Collision,Side Collision,Total Loss,Other,OH,Hillsdale,1643 Washington Hwy,20,3,Unknown,0,1,,11/1/2025,2/20/2024,Volkswagen,Passat,2021,Gray,15144.37,5982,3890,1,N,0,24527.38,7224.79,3074.12,14228.47,UX39O9355,No
AA00000006,8/1/2022,17,32,189063232,IL,11/6/2023,5/6/2024,500/1000,500,1592.41,0,474324,MALE,Masters,prof-specialty,yachting,husband,58900,-29100,No,2/6/2024,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Columbus,5771 Best St,22,1,Unknown,2,3,YES,12/16/2025,2/10/2024,Jeep,Wrangler,2024,Black,23248.36,8541,6745,0,N,0,2675.78,112.04,2493.54,70.2,XK41H9013,No
AA00000007,7/1/2022,18,40,145356955,IN,10/16/2023,4/16/2024,100/300,500,1463.95,0,430567,FEMALE,JD,sales,skydiving,own-child,0,0,No,2/11/2024,Multi-vehicle Collision,Unknown,Total Loss,Police,NC,Springfield,4545 4th Ridge,20,3,Unknown,0,0,YES,3/9/2026,2/11/2024,Dodge,Neon,2017,Black,7658.39,17466,6836,0,N,0,19160.58,4890.49,3242.39,11027.7,MY30O4303,No
AA00000008,2/1/2022,23,32,193384481,IN,10/14/2023,4/14/2024,250/500,1000,988.93,0,614187,FEMALE,High School,craft-repair,golf,unmarried,27600,0,No,1/23/2024,Single Vehicle Collision,Front Collision,Total Loss,Other,NY,Columbus,2889 Francis St,11,1,Unknown,2,3,YES,10/30/2025,1/26/2024,Dodge,Neon,2021,Silver,14485.73,9425,7446,0,N,0,5644.45,470.3,317.06,4857.09,II74U8175,No
AA00000009,11/1/2022,14,32,120145403,IL,8/26/2023,2/26/2024,500/1000,500,1612.43,0,456762,FEMALE,MD,other-service,yachting,own-child,36400,0,No,1/8/2024,Single Vehicle Collision,Side Collision,Total Loss,Fire,VA,Springfield,2087 Apache Ave,2,1,Unknown,2,1,YES,6/11/2025,1/11/2024,BMW,3 Series,2024,Black,23718.36,8038,6603,0,N,0,17963.3,3941.9,7210.56,6810.84,UD66G3260,No
AA00000010,5/1/2022,20,40,133676496,IL,10/5/2023,4/5/2024,500/1000,1000,1361.45,0,604833,MALE,PhD,handlers-cleaners,camping,unmarried,39300,0,No,2/15/2024,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,OH,Northbend,7570 Cherokee Drive,12,1,YES,0,2,YES,11/12/2025,2/19/2024,Suburu,Forrestor,2021,Gray,15285.62,6344,4353,1,N,0,4589.87,1423.96,362.22,2803.69,QM79R9596,No
AA00000011,10/1/2022,15,61,199046488,OH,11/15/2023,5/15/2024,100/300,2000,1137.16,0,615561,FEMALE,High School,exec-managerial,skydiving,other-relative,0,-51000,No,2/15/2024,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Springfield,5971 5th Hwy,21,3,YES,1,2,YES,11/26/2025,2/19/2024,Audi,A3,2015,White,5489.19,7343,5411,1,N,0,20467.14,15484.03,4867.4,115.71,ME48T1151,No
AA00000012,11/1/2022,14,44,140353487,OH,9/12/2023,3/12/2024,100/300,1000,1280.88,0,433981,MALE,MD,other-service,basketball,other-relative,59400,-32200,No,2/6/2024,Single Vehicle Collision,Rear Collision,Total Loss,Other,WV,Riverwood,5312 Francis Ridge,21,1,NO,0,1,NO,10/16/2025,2/8/2024,Accura,TL,2018,Blue,8729.61,8528,6296,0,N,1,762.64,143.25,301.99,317.4,QW06O4912,No
AA00000013,1/1/2023,12,30,135534304,IN,9/14/2023,3/14/2024,500/1000,500,951.46,0,467227,MALE,JD,handlers-cleaners,golf,not-in-family,0,-35500,No,1/31/2024,Multi-vehicle Collision,Front Collision,Major Damage,Other,WV,Riverwood,1328 Texas Lane,8,3,NO,0,3,,6/6/2025,2/5/2024,Volkswagen,Passat,2016,Black,6476.02,10188,7412,0,Y,0,17693.05,6860.5,4982.68,5849.87,KF93S7431,No
AA00000014,8/1/2022,17,43,136907088,IL,8/28/2023,2/28/2024,100/300,500,1508.12,6000000,433275,MALE,PhD,craft-repair,basketball,wife,0,0,No,1/20/2024,Multi-vehicle Collision,Rear Collision,Major Damage,Other,NY,Columbus,5276 2nd Lane,0,3,Unknown,2,1,NO,4/22/2025,1/23/2024,Jeep,Wrangler,2024,Gray,23269.89,14077,3038,0,Y,0,20735.51,1295.32,10165.68,9274.51,RX98J8076,No
AA00000015,11/1/2022,14,30,173927158,OH,11/4/2023,5/4/2024,100/300,500,1399.27,6000000,448913,MALE,College,prof-specialty,hiking,wife,38900,-48700,No,2/24/2024,Single Vehicle Collision,Side Collision,Total Loss,Fire,NC,Arlington,9417 Tree Hwy,22,1,Unknown,0,0,YES,7/3/2025,2/24/2024,Ford,Escape,2016,Blue,6474.08,8375,3484,0,N,0,4432.05,1613.69,2563.84,254.52,VV72O7327,No
AA00000016,9/1/2022,16,55,192848218,IN,10/18/2023,4/18/2024,250/500,500,1607.36,0,444626,MALE,MD,sales,yachting,own-child,0,0,No,1/19/2024,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Arlington,8624 Francis Ave,21,4,Unknown,0,2,NO,8/28/2025,1/23/2024,Dodge,RAM,2016,Red,8297.84,14929,7511,0,Y,0,8510.54,1881.99,3002.59,3625.96,WP00J9804,No
AA00000017,12/1/2022,13,34,176785687,IN,9/4/2023,3/4/2024,100/300,500,1706.79,0,462479,MALE,Masters,protective-serv,dancing,other-relative,0,0,No,1/7/2024,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,OH,Hillsdale,9706 MLK Lane,1,1,NO,1,1,YES,6/3/2025,1/9/2024,Honda,Civic,2022,White,15155.97,13759,5477,0,N,0,8835.12,1551.91,1578.68,5704.53,GR59S1093,No
AA00000018,2/1/2022,23,46,136881491,OH,9/19/2023,3/19/2024,500/1000,1000,1372.18,0,439929,MALE,High School,exec-managerial,bungie-jumping,not-in-family,0,0,No,1/13/2024,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Riverwood,5914 Oak Ave,22,3,NO,1,3,,8/28/2025,1/18/2024,BMW,X5,2018,Gray,19307.93,8149,4534,0,N,0,7610.52,3036.7,1315.4,3258.42,WN85C0904,No
AA00000019,1/1/2023,12,29,118535855,IL,9/17/2023,3/17/2024,250/500,1000,1726.91,0,456570,MALE,High School,other-service,basketball,own-child,0,0,No,1/28/2024,Vehicle Theft,Unknown,Trivial Damage,,VA,Hillsdale,4453 Best Ave,14,1,Unknown,0,0,,10/8/2025,1/31/2024,Audi,A5,2021,Black,13971.42,13289,7446,0,N,0,11521.61,716.86,6801.17,4003.58,VT46U2466,No
AA00000020,9/1/2022,16,43,178634265,IN,10/"""

file02_content = """Claim_ID,Bind_Date1,Customer_Life_Value1,Age_Insured,Policy_Num,Policy_State,Policy_Start_Date,Policy_Expiry_Date,Policy_BI,Policy_Ded,Policy_Premium,Umbrella_Limit,Insured_Zip,Gender,Education,Occupation,Hobbies,Insured_Relationship,Capital_Gains,Capital_Loss,Garage_Location,Accident_Date,Accident_Type,Collision_Type,Accident_Severity,authorities_contacted,Acccident_State,Acccident_City,Accident_Location,Accident_Hour,Num_of_Vehicles_Involved,Property_Damage,Bodily_Injuries,Witnesses,Police_Report,DL_Expiry_Date,Claims_Date,Auto_Make,Auto_Model,Auto_Year,Vehicle_Color,Vehicle_Cost,Annual_Mileage,DiffIN_Mileage,Low_Mileage_Discount,Fraud_Ind,Commute_Discount,Total_Claim,Injury_Claim,Property_Claim,Vehicle_Claim,Vehicle_Registration,Check_Point
BB00000001,11/1/2022,14,35,155203481,IN,10/12/2023,4/12/2024,500/1000,2000,1123.89,0,468313,MALE,MD,priv-house-serv,video-games,unmarried,35400,-49200,No,1/21/2024,Multi-vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,4119 Texas St,0,3,Unknown,1,3,,3/31/2026,1/23/2024,Suburu,Forrestor,2024,Black,23039.29,8873,5683,0,N,0,6208.92,787.12,4184.84,1236.96,JP09S1996,No
BB00000002,1/1/2022,24,30,189485666,OH,9/14/2023,3/14/2024,500/1000,1000,1166.54,0,479852,FEMALE,Masters,prof-specialty,sleeping,not-in-family,47700,-59300,No,1/21/2024,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Arlington,9316 Pine Ave,3,1,YES,2,0,NO,6/23/2025,1/26/2024,Dodge,Neon,2023,Silver,20142.33,14368,6695,0,Y,0,15441.4,633.67,7970.99,6836.74,JY94J4803,No
BB00000003,11/1/2022,14,52,145849195,IL,10/30/2023,4/30/2024,250/500,1000,1242.96,7000000,449800,FEMALE,High School,other-service,paintball,own-child,0,-37100,No,2/22/2024,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Riverwood,5838 Pine Lane,2,2,YES,2,0,,9/10/2025,2/27/2024,Accura,RSX,2019,White,10753.78,13889,7130,0,N,0,13560.5,225.07,11033.06,2302.37,HU95F8182,No
BB00000004,11/1/2022,14,39,183287405,OH,8/19/2023,2/19/2024,250/500,1000,1351.1,0,478456,FEMALE,PhD,tech-support,bungie-jumping,unmarried,0,0,No,1/2/2024,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Arlington,8973 Washington St,19,3,NO,0,2,NO,12/29/2025,1/2/2024,Saab,95,2017,Blue,7795.04,15671,2791,0,Y,0,2777.97,874.48,671.78,1231.71,ME57A7253,No
BB00000005,12/1/2022,13,35,180131023,IL,9/25/2023,3/25/2024,100/300,2000,1219.04,0,456602,MALE,MD,handlers-cleaners,paintball,own-child,63600,-68700,No,1/11/2024,Multi-vehicle Collision,Rear Collision,Major Damage,Police,OH,Hillsdale,7380 5th Hwy,19,3,Unknown,2,0,NO,7/7/2025,1/15/2024,Nissan,Pathfinder,2021,Red,14389.54,9423,7552,0,N,0,3455.36,23.65,3138.9,292.81,AH97U6339,No
BB00000006,7/1/2022,18,42,164466083,OH,10/1/2023,4/1/2024,250/500,500,1337.56,0,605141,FEMALE,College,prof-specialty,video-games,unmarried,0,0,No,1/12/2024,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Riverwood,5211 Weaver Drive,18,1,Unknown,1,2,YES,7/29/2025,1/15/2024,Dodge,RAM,2019,Black,13246.55,11753,7205,0,N,0,15813.69,11265.88,861.56,3686.25,TW77L1507,No
BB00000007,10/1/2022,15,40,186497234,OH,9/8/2023,3/8/2024,100/300,2000,1291.7,0,609837,FEMALE,JD,sales,kayaking,not-in-family,0,-55600,No,1/8/2024,Single Vehicle Collision,Side Collision,Minor Damage,Other,SC,Northbend,3220 Rock Drive,21,1,NO,1,0,YES,5/27/2025,1/10/2024,Dodge,Neon,2018,Red,8857.3,8565,6991,0,N,0,7335.9,3454.59,2797.91,1083.4,YZ27C6648,No
BB00000008,9/1/2022,16,40,158942066,OH,7/4/2023,1/4/2024,500/1000,2000,1217.69,0,440106,FEMALE,MD,prof-specialty,reading,wife,24000,0,No,1/26/2024,Multi-vehicle Collision,Side Collision,Major Damage,Other,SC,Springfield,9138 1st St,6,3,Unknown,1,0,,6/9/2025,1/28/2024,Dodge,Neon,2016,Silver,6528.26,5916,6197,1,Y,0,19812.78,5528.99,13166.73,1117.06,HR96D2335,No
BB00000009,11/1/2022,14,39,189881960,OH,11/19/2023,5/19/2024,250/500,1000,1393.57,0,478423,MALE,PhD,machine-op-inspct,movies,not-in-family,47600,-39600,No,2/27/2024,Parked Car,Unknown,Minor Damage,Police,VA,Northbend,5821 2nd St,5,1,NO,0,1,YES,12/31/2025,3/2/2024,Ford,F150,2018,White,10727.21,13493,4301,0,N,0,12994.75,3880.28,2935.69,6178.78,UT55T9418"""

file03_content = """Claim_ID,Bind_Date1,Customer_Life_Value1,Age_Insured,Policy_Num,Policy_State,Policy_Start_Date,Policy_Expiry_Date,Policy_BI,Policy_Ded,Policy_Premium,Umbrella_Limit,Insured_Zip,Gender,Education,Occupation,Hobbies,Insured_Relationship,Capital_Gains,Capital_Loss,Garage_Location,Accident_Date,Accident_Type,Collision_Type,Accident_Severity,authorities_contacted,Acccident_State,Acccident_City,Accident_Location,Accident_Hour,Num_of_Vehicles_Involved,Property_Damage,Bodily_Injuries,Witnesses,Police_Report,DL_Expiry_Date,Claims_Date,Auto_Make,Auto_Model,Auto_Year,Vehicle_Color,Vehicle_Cost,Annual_Mileage,DiffIN_Mileage,Low_Mileage_Discount,Commute_Discount,Total_Claim,Injury_Claim,Property_Claim,Vehicle_Claim,Vehicle_Registration,Check_Point
CC00000001,3/1/2022,22,31,116375016,IN,10/4/2023,4/4/2024,500/1000,500,708.64,6000000,470610,MALE,High School,machine-op-inspct,reading,unmarried,53500,0,No,1/6/2024,Single Vehicle Collision,Side Collision,Total Loss,Police,WV,Northbend,4546 Tree St,9,1,NO,0,2,YES,3/21/2025,1/10/2024,Suburu,Legacy,2024,White,22656.54,17338,5441,0,0,10048.28,1219,4153.83,4675.45,VG98O2447,No
CC00000002,9/1/2022,16,34,189560103,IL,10/21/2023,4/21/2024,100/300,2000,1057.77,0,477382,FEMALE,JD,tech-support,bungie-jumping,unmarried,0,-57700,No,2/16/2024,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,NC,Riverwood,2003 Maple Hwy,22,3,Unknown,1,1,NO,9/1/2025,2/17/2024,Audi,A3,2018,Gray,9290.97,9809,2825,0,0,12668.14,5979.67,1043.39,5645.08,FN33R3556,No
CC00000003,10/1/2022,15,27,197359502,IL,10/20/2023,4/20/2024,100/300,1000,1384.51,0,476727,MALE,PhD,adm-clerical,reading,not-in-family,13100,-38200,No,2/18/2024,Single Vehicle Collision,Rear Collision,Major Damage,Other,SC,Springfield,9484 Pine Drive,14,1,NO,2,3,,2/25/2026,2/19/2024,Volkswagen,Jetta,2016,Gray,6675.48,15107,5482,0,0,802.77,372.55,82.28,347.94,VI12A3028,No
CC00000004,1/1/2022,24,28,123790687,OH,10/13/2023,4/13/2024,500/1000,1000,986.53,0,472720,FEMALE,High School,adm-clerical,polo,other-relative,62700,0,No,2/16/2024,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,VA,Columbus,7819 Oak St,11,1,NO,0,2,YES,8/12/2025,2/17/2024,Honda,CRV,2016,Gray,6199.24,12240,4943,0,0,11316.82,1797.4,5908.36,3611.06,QY82Q7886,No"""


# Load the datasets
df1 = pd.read_csv(io.StringIO(file01_content))
df2 = pd.read_csv(io.StringIO(file02_content))
df3 = pd.read_csv(io.StringIO(file03_content))

# Combine the dataframes
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Correct potential column inconsistencies due to missing 'Fraud_Ind' in df3 header
if 'Fraud_Ind' not in combined_df.columns:
    combined_df['Fraud_Ind'] = None

# --- Data Preprocessing (Re-running from previous step to ensure data is prepared) ---

# 1. Date Conversion
date_cols = ['Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date', 'Accident_Date', 'DL_Expiry_Date', 'Claims_Date']
for col in date_cols:
    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')

# 2. Handle Missing Values
numerical_cols = combined_df.select_dtypes(include=['number']).columns
categorical_cols = combined_df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

for col in categorical_cols:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

combined_df['Property_Damage'] = combined_df['Property_Damage'].replace('Unknown', combined_df['Property_Damage'].mode()[0])

# 3. Feature Engineering
for col in date_cols:
    if col in combined_df.columns and pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        combined_df[col + '_Year'] = combined_df[col].dt.year
        combined_df[col + '_Month'] = combined_df[col].dt.month
        combined_df[col + '_Day'] = combined_df[col].dt.day

combined_df['Vehicle_Age_at_Accident'] = combined_df['Accident_Date_Year'] - combined_df['Auto_Year']
combined_df['Vehicle_Age_at_Accident'] = combined_df['Vehicle_Age_at_Accident'].fillna(combined_df['Vehicle_Age_at_Accident'].median())

combined_df['Days_Policy_to_Accident'] = (combined_df['Accident_Date'] - combined_df['Policy_Start_Date']).dt.days
combined_df['Days_Policy_to_Accident'] = combined_df['Days_Policy_to_Accident'].fillna(combined_df['Days_Policy_to_Accident'].median())

# 4. Target Variable Encoding
combined_df['Fraud_Ind'] = combined_df['Fraud_Ind'].map({'Y': 1, 'N': 0})
if combined_df['Fraud_Ind'].isnull().any():
    combined_df['Fraud_Ind'] = combined_df['Fraud_Ind'].fillna(combined_df['Fraud_Ind'].mode()[0])

# 5. Remove Original Date Columns and other identifier/less useful columns
cols_to_drop = date_cols + ['Claim_ID', 'Policy_Num', 'Vehicle_Registration', 'Check_Point']
combined_df = combined_df.drop(columns=[col for col in cols_to_drop if col in combined_df.columns])

# 6. Categorical Encoding
categorical_cols_after_drop = combined_df.select_dtypes(include=['object']).columns
combined_df = pd.get_dummies(combined_df, columns=categorical_cols_after_drop, drop_first=True)

# --- Model Training and Evaluation ---

# Separate features (X) and target (y)
X = combined_df.drop('Fraud_Ind', axis=1)
y = combined_df['Fraud_Ind']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for ROC AUC

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Calculate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)


print("\n--- Model Training and Evaluation Complete ---")
print(f"Model Used: Random Forest Classifier")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(conf_matrix)

# --- Visualizations ---

# Plot Confusion Matrix
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cmp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
cmp.plot(ax=ax_cm, cmap='Blues')
ax_cm.set_title('Confusion Matrix')
plt.show() # Display the plot

# Plot Feature Importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 7))
feature_importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
plt.show() # Display the plot

print("\nTop 10 Feature Importances:\n")
print(feature_importances.head(10))