import psycopg2

class PostgresqlUtil(object):

   def __init__(self):
      pgConnectString2 = "host='localhost' port='5432' dbname='herohe' user='robert' password='fenris'"
      pgConnection = psycopg2.connect(pgConnectString2)
      pgCursor = pgConnection.cursor()

   def saveHistogram(self, slidename, tilingx, tilingy, tilex, tiley, histogram):
      self.pgCursor()
      print("Implement Save Histogram!")