import time

import dataiku
import numpy as np
import pandas as pd

from cashflow import PERIOD_DAYS, cutoff_date, read_dataset, write_daily
from forecast import active_model_id
from warehouse import CATEGORY_OF_TYPE, OTHER

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
cutoff = cutoff_date()
print(f"Production model {active_model_id()}, cut-off day {cutoff:%Y-%m-%d}")   # stops here if there is none yet
