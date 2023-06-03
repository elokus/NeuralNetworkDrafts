import pandas as pd
import custom_methods
dat = pd.read_csv('DEXUSEU.csv', parse_dates=[
                  'DATE'])
dat['DEXUSEU'] = pd.to_numeric(dat['DEXUSEU'], errors='coerce')
# dat = dat.dropna(how='any', axis=0)

data = custom_methods.cust(dat)
# data = data.clean_data()
data = data.outlier_removal(var="DEXUSEU")
print(data)
