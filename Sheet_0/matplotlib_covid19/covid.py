from urllib.request import urlopen
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import csv

def extract_data(reader, countries):
    result = {c: [] for c in countries}
    for row in reader:
        country = row["countriesAndTerritories"]
        if country in countries:
            result[country].append(
                [
                    date(int(row["year"]), int(row["month"]), int(row["day"])),  # Date
                    int(row["cases"]),  # New Cases on that day
                ]
            )

    return result


# Load Covid-19 data
countries = ["Palestine", "Jordan", "Germany", "China", "United_States_of_America"]

with urlopen("http://opendata.ecdc.europa.eu/covid19/casedistribution/csv") as f:
    lines = [line.decode("utf-8") for line in f.readlines()]
reader = csv.DictReader(lines)
data = extract_data(reader, countries)

# `data` looks like this:
# {
#     'Germany': [
#         [date(2020, 1, 1), 10],
#         [date(2020, 1, 2), 20],
#         ...
#     ],
#     'United_States_of_America': [
#         [date(2020, 1, 1), 100],
#         [date(2020, 1, 2), 200],
#         ...
#     ],
#     'China': [
#         [date(2020, 1, 1), 1000],
#         [date(2020, 1, 2), 2000],
#         ...
#     ],
#     ...
# }


# Compute total cases for each day
# Ignore people who have recovered or died
plot_data = {k: [] for k in data.keys()}

for country, values in data.items():
    # remember: each value is a list of shape [date, cases]
    cumsum = 0
    # Sort by date in ascending order and sum up the cases
    for day in sorted(values, key=lambda x: x[0]):
        cumsum += day[1]  # day[1] is the number of cases on that day
        day.append(cumsum)  # this will give: [date, cases, cumsum]

        # Create Data, starting from the day where
        # 100 people have been infected
        if cumsum >= 100:
            plot_data[country].append(cumsum)


# `plot_data` looks like this:
# {
#     'Germany': [100, 200, 300, ...],
#     'United_States_of_America': [100, 200, 300, ...],
#     'China': [100, 200, 300, ...],
#     ...
# }


# hypothetical curve where total cases double
# every five days
def exp_double_time(x, start_cases=100, double_time=5):
    return start_cases * np.power(np.power(2, 1 / double_time), x)  # 100 * (2^(1/5))^x


# Plot everything in a log-plot with legend
def display_name(country):
    return country.replace("_", " ")


x = np.arange(start=0, stop=max(len(v) for v in plot_data.values()))
fig = plt.figure(figsize=(12, 9))  # width, height in inches

plt.title("Total SARS-CoV-2 Cases per Country")
plt.xlabel("Days since 100th case")
plt.ylabel("Total SARS-CoV-2 cases")
plt.yscale("log")  # log scale for y-axis (total cases)

for country, values in plot_data.items():
    plt.plot(x[: len(values)], values, label=display_name(country))

# Plot hypothetical curve
plt.plot(x, exp_double_time(x), label="Cases double every 5 days", linestyle="dashed")

plt.legend()
plt.grid(linestyle='dashed')
plt.show()
plt.close(fig)

