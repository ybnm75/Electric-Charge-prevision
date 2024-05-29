import pandas as pd

# Assuming 'data' is your list of rows as described
data = [
    ["2013-01-10 16:00:00",0.4408571428571428,0.4441428571428571,0.4585,0.4497857142857143,0.4297857142857142,0.4512142857142857,0,0,0,1,0,0,0,0],
    ["2013-01-10 18:00:00",0.5680714285714286,0.4817857142857142,0.566,0.4408571428571428,0.5747857142857142,0.4441428571428571,0,0,0,1,0,0,0,0],
    ["2013-01-10 22:00:00",0.4933571428571429,0.5402857142857143,0.4940714285714285,0.5701428571428572,0.5020714285714286,0.5818571428571429,0,0,0,1,0,0,0,0],
    ["2013-01-11 16:00:00",0.4047857142857143,0.3932142857142857,0.4097857142857143,0.3914285714285714,0.4408571428571428,0.3940714285714286,0,0,0,0,1,0,0,1]
    # Add more rows...
]

# Create DataFrame
df = pd.DataFrame(data)

# Rename columns using the first row
df.columns = df.iloc[0]

# Set index to DateTime column
df.index = pd.to_datetime(df.iloc[:, 0])  # Set index using the first column which contains the date-time values
df = df.drop(columns=df.columns[0])  # Drop the first column as it's now the index

# Function to get the next 24 hours' consumption
def get_next_24_hours_consumption(consumption_value, datetime):
    start_date = datetime
    end_date = datetime + pd.Timedelta(hours=24)
    last_column_name = df.columns[-1]  # Get the name of the last column (assuming it's the output column)
    consumption_data = df.loc[start_date:end_date, last_column_name]
    # Get the output consumption value for the final time stamp of the 24-hour period
    final_consumption_value = consumption_data.iloc[-1]
    return final_consumption_value

# Example usage
example_date = pd.to_datetime("2013-01-10 16:00:00")
result = get_next_24_hours_consumption(0.4408571428571428, example_date)
print(result)
