import pandas as pd
import numpy as np


df = pd.read_csv("resume_job_dataset.csv")

new_job_desc = []
new_resume = []

for jd, res, score in zip(df["job_description"], df["resume"], df["match_score"]):
    if score <= 2:
        job_exp = np.random.randint(3, 8)   
        res_exp = np.random.randint(0, 2)   
    elif score == 3:
        job_exp = np.random.randint(2, 6)
        res_exp = np.random.randint(1, 4)   
    elif score == 4:
        job_exp = np.random.randint(2, 6)
        res_exp = job_exp + np.random.choice([-1, 0, 1])  
    else: 
        job_exp = np.random.randint(3, 8)
        res_exp = job_exp +  np.random.randint(0, 2)

   
    jd_new = jd + f" Requires {job_exp}+ years of relevant experience."

   
    res_new = res + f" Has {res_exp}+ years of work experience."

    new_job_desc.append(jd_new)
    new_resume.append(res_new)


df["job_description"] = new_job_desc
df["resume"] = new_resume


df.to_csv("resume_job_dataset_with_exp_text.csv", index=False)

print(" Updated dataset saved as resume_job_dataset_with_exp_text.csv")

