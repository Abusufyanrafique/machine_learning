import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# df =sns.load_dataset("tips")
# print(df)
# sns.scatterplot(x=df["total_bill"], y=df["tip"], hue=df["sex"],style= df["smoker"], data=df, )


df =sns.load_dataset("titanic")
print(df)
# sns.barplot(x="pclass", y="fare",hue=df["sex"], data=df)
# pandas profiling 
# profile = ProfileReport(df, explorative=True)

# Save report as an HTML file
profile.to_file("EDA_Report.html")

print("EDA Report Generated: Open 'EDA_Report.html' in your browser.")
sns.boxplot(x="sex", y="age", data=df)

# df =sns.load_dataset("flights")
# df =sns.load_dataset("iris")
plt.show()