import time

import dataiku
import numpy as np
import pandas as pd

from cashflow import cutoff_month, read_dataset
from forecast import active_model_id
from warehouse import CATEGORY_OF_TYPE, OTHER

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
cutoff = cutoff_month()
print(f"Production model {active_model_id()}, cut-off month {cutoff:%Y-%m}")   # stops here if there is none yet
