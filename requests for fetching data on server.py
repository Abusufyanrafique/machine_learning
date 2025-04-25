import requests
from io import StringIO

url = "https://example.com/data.csv"
response = requests.get(url, headers={"Authorization": "Bearer YOUR_TOKEN"})

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    print(df.head())
else:
    print("Failed to fetch the file:", response.status_code)
