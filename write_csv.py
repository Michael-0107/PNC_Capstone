import csv
import pandas as pd

import csv



### Build base csv from temp.txt
# q = {
#     1: ("-01-01", "-03-31"),
#     2: ("-04-01", "-06-30"),
#     3: ("-07-01", "-09-30"),
#     4: ("-10-01", "-12-31")
#     }

# year = list(range(2009,2018))

# with open('temp.txt', 'r') as in_file:
#     stripped = (line.strip() for line in in_file)
#     lines = [line.split("\t")[:3] for line in stripped if line]
#     new_lines = []
#     for line in lines:
#         for y in year:
#             for quar in q:
#                 new_line = line.copy()
#                 new_line.append(str(y)+q[quar][0])
#                 new_line.append(str(y)+q[quar][1])
#                 new_lines.append(new_line)
        
#     with open('comp.csv', 'w') as out_file:
#         writer = csv.writer(out_file)
#         writer.writerow(('number', 'ticker', 'company', 'startdate', 'enddate', 'news_summary'))
#         writer.writerows(new_lines)