# Explore Popularity of FordGoBike
## by Sahn


## Datasets

The project uses 6 FordGoBike trip datasets. Data contain trip data between January 2019 and June 2019.Each month has about 200,000 trip data.

Data is downloaded from [Lyft System Data](https://www.lyft.com/bikes/bay-wheels/system-data).

Raw files are:
- 201901-fordgobike-tripdata.csv
- 201902-fordgobike-tripdata.csv
- 201903-fordgobike-tripdata.csv
- 201904-fordgobike-tripdata.csv
- 201905-fordgobike-tripdata.csv
- 201906-fordgobike-tripdata.csv


## Usage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
import warnings
warnings.simplefilter("ignore")
```


## Summary of Findings

- More males use FordGoBike then other genders.
- Most of FordGoBike riders are subscribers.
- Majority of FordGoBike riders' birth years are in 1980's or 1990's. Ratio of older rider is higher in the morning and ratio of younger riders is higher in the evening
- Riders ride FordGoBike in March the most. Similar to Month, both start and end data show two picks in the time of the day: one in the morning (~ 8 AM) and one in the late afternoon (~ 5 PM).
- Data is collected from three cities, and San Francisco uses FordGoBike the most.
- Majority of FordGoBike rider's trip duration is less than 200 minutes.
- From the trip start time, end time, and duration, it is possible to say that longer trip duration is caused when riders keep the bike overnight.


## Key Insights for Presentation

> FordGoBike is most popular for males who were born in 1980's or 1990's and live (or work) in San Francisco


## Readme Template Reference
[Make a Readme](https://www.makeareadme.com/)
