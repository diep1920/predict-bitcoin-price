import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import sys
import time


def read_params(fname):
	params = {}
	fR = open(fname, 'r')
	for line in fR:
		print(line.strip())
		key, val = line.strip().split('=')
		key = str(key.strip())
		val = str(val.strip())
		params[key] = str(val)

	fR.close()
	return params


#input : ticker_pair. exp: XBTUSD for Bitcoin - USD
#output: a dataframe row of columns ['ask_price', 
#									 'ask_whole_lot_volume',
# 									 'ask_lot_volume',  
#									 'bid_price',
#									 'bid_whole_lot_volume',
#									 'bit_lot_volume',
#									 'close_price',
#									 'close_volume'] 
#		at instant
def get_ticker(ticker_pair):
	ticker = k.get_ticker_information(ticker_pair)

	dict_data = {'ask_price' : ticker['a'][0][0],
				 'ask_whole_lot_volume' : ticker['a'][0][1],
				 'ask_lot_volume' : ticker['a'][0][2],
				 'bid_price' : ticker['b'][0][0],
				 'bid_whole_lot_volume' : ticker['b'][0][1],
				 'bit_lot_volume' : ticker['b'][0][2],
				 'close_price' : ticker['c'][0][0],
				 'close_volume' : ticker['c'][0][1]}

	return dict_data


api = krakenex.API()
k = KrakenAPI(api)

params = read_params(sys.argv[1])
pairs = params['pairs']
interval = params['interval']
no_of_records = params['no_of_records']
file_name = params['name']

df = pd.DataFrame(columns = ['ask_price', 'ask_whole_lot_volume', 'ask_lot_volume', \
                             'bid_price', 'bid_whole_lot_volume', 'bit_lot_volume', \
                             'close_price', 'close_volume'                          \
                            ])
dict_list = []
for i in range(0, int(no_of_records)):
	new_row = get_ticker(pairs)
	dict_list.append(new_row)
	time.sleep(int(interval))

df = pd.DataFrame.from_dict(dict_list)
df.to_csv('./dataset/' + file_name + '.csv', index=False)