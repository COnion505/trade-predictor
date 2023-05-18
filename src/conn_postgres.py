import psycopg2
import codecs
from datetime import datetime as dt
import time
from decimal import Decimal
HOSTNAME = 'localhost'
PORT = '5432'
USER = 'postgres'
PASSWORD = 'Noihs505'
DB_NAME = 'fx'
# DB接続関数 
def get_connection(): 
    return psycopg2.connect('postgresql://{user}:{password}@{host}:{port}/{dbname}'
        .format( 
            user=USER, password=PASSWORD, host=HOSTNAME, port=PORT, dbname=DB_NAME 
        ))
def update_local_time():
	start = time.perf_counter()
	print('start process...')
	print('connecting database...')
	conn = get_connection()
	cur = conn.cursor()
	print('done.')
	print('get data...')
	#cur.execute('SELECT \"LOCAL_TIME\" FROM usdjpy WHERE \"LOCAL_TIME\" LIKE \'20.04.2020%\'')
	cur.execute('SELECT \"LOCAL_TIME\" FROM usdjpy')
	rows = cur.fetchall()
	print('done.')
	#f = codecs.open('test.txt','w','utf-8')
	i = 0
	rows_count = len(rows)
	print('row counts:{}'.format(rows_count))
	print('update fetched rows...')
	#time.sleep(3)
	for r in rows:
		#print('{}/{}'.format(i+1,rows_count))
		i += 1
		if i%100000 == 0:
			print('log: done {} rows.'.format(i))
		s = datetime.datetime.strptime(r[0], '%d.%m.%Y-%H:%M:%S.%f')
		#print('UPDATE usdjpy SET \"LOCAL_TIME\" = \'{}\' WHERE \"LOCAL_TIME\" = \'{}\''.format(s,r[0]), file=f)
		cur.execute('UPDATE usdjpy SET \"LOCAL_TIME\" = \'{}\' WHERE \"LOCAL_TIME\" = \'{}\''.format(s,r[0]))
		#print(r,file=f)
	#f.close()
	conn.commit()
	print('done.')
	cur.close()
	conn.close()

	end = time.perf_counter()
	elaspe = int(end - start)
	print('end with {} seconds'.format(elaspe))

def get_dayOfTheWeek():
	conn = get_connection()
	cur = conn.cursor()
	#cur.execute('SELECT \"LOCAL_TIME\" FROM usdjpy')
	cur.execute('SELECT \"LOCAL_TIME\" FROM usdjpy WHERE \"LOCAL_TIME\" LIKE \'2020-02%\'')
	rows = cur.fetchall()
	f = codecs.open('test.txt','w','utf-8')
	i = 0
	print('start')
	for r in rows:
		i += 1
		if i%100000 == 0:
			print('log: done {} rows.'.format(i))
		d = dt.strptime(r[0], '%Y-%m-%d %H:%M:%S')
		#print('{} -> {},{}'.format(d,d.strftime('%A'),d.weekday()), file=f)
		#s = 'UPDATE usdjpy SET \"WEEKDAYS\" = \'{}\' WHERE \"LOCAL_TIME\" = \'{}\''.format(d.weekday(),r[0])
		s = 'INSERT INTO WEEKDAYS (\"NO\",\"WEEKDAY\") VALUES (\'{}\',{})'.format(d.strftime('%A'),d.weekday())
		print(s, file=f)
		#print(s, file=f)
		#cur.execute(s)
	f.close()
	print('done')
	conn.commit()
	cur.close()
	conn.close()

def get_data():
	conn = get_connection()
	cur = conn.cursor()
	#sql = 'SELECT \"LOCAL_TIME\", \"CLOSE\" FROM usdjpy WHERE \"LOCAL_TIME\" LIKE \'202%\' AND \"WEEKDAY\" NOT IN (\'5\',\'6\') ORDER BY \"LOCAL_TIME\"'
	sql = 'SELECT \"LOCAL_TIME\", \"CLOSE\" FROM usdjpy WHERE \"VOLUME\" <> \'0\' AND \"LOCAL_TIME\" LIKE \'2020-01%\' ORDER BY \"LOCAL_TIME\"'
	cur.execute(sql)
	rows = cur.fetchall()
	f= codecs.open('test2.csv', 'w', 'utf-8')
	print('LOCAL_TIME,CLOSE,GROUP,GAP', file=f)
	#for r in rows:
		#s = '{},{}'.format(r[0],r[1])
		#print(s, file=f)
	print(len(rows))
	for i in range(0,len(rows)-1):
		group = 'NULL'
		gap =0
		if i < len(rows)-1-60:
			future_close = Decimal(rows[i+60][1])
			current_close = Decimal(rows[i][1])
			gap = future_close - current_close
			if gap < -0.050:
				group = '-1'
			elif gap < 0.050:
				group='0'
			else:
				group ='1'
		print('{},{},{},{}'.format(rows[i][0],rows[i][1],group,gap), file=f)
	f.close()
	cur.close()
	conn.close()
get_data()

#get_dayOfTheWeek()
