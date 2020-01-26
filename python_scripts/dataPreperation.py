import os
import psycopg2

pgConnectString = "host='localhost' port='5432' dbname='herohe' user='robert' password='fenris'"
pgConnection=psycopg2.connect(pgConnectString)
pgCursor = pgConnection.cursor()

def dropTables():
    query_drop_images = "DROP TABLE images;"
    query_drop_image_parameter = "DROP TABLE image_parameter;"
    query_drop_herohe_data = "DROP TABLE herohe_data;"
    query_drop_lerndata = "DROP TABLE lerndata;"
    pgCursor.execute(query_drop_images)
    pgCursor.execute(query_drop_image_parameter)
    pgCursor.execute(query_drop_herohe_data)
    pgCursor.execute(query_drop_lerndata)
    pgConnection.commit()

def createTables():
    query_create_images = "CREATE TABLE images (" + \
                            "id serial," + \
                            "name character varying," + \
                            "her2status character varying," + \
                            "cell_count integer" + \
                            ");"
    query_create_image_parameter = "CREATE TABLE image_parameter (" + \
                                   "id bigserial NOT NULL, " + \
                                   "image_id integer, " + \
                                   "filename character varying, " + \
                                   "import_filename character varying, " + \
                                   "name character varying, " + \
                                   "class character varying, " + \
                                   "parent character varying, " + \
                                   "roi character varying, " + \
                                   "centroid_x_um numeric, " + \
                                   "centroid_y_um numeric, " + \
                                   "nucleus_area numeric, " + \
                                   "nucleus_perimeter numeric, " + \
                                   "nucleus_circularity numeric, " + \
                                   "nucleus_max_caliper numeric, " + \
                                   "nucleus_min_caliper numeric, " + \
                                   "nucleus_eccentricity numeric, " + \
                                   "nucleus_hematoxylin_od_mean numeric, " + \
                                   "nucleus_hematoxylin_od_sum numeric, " + \
                                   "nucleus_hematoxylin_od_std_dev numeric, " + \
                                   "nucleus_hematoxylin_od_max numeric, " + \
                                   "nucleus_hematoxylin_od_min numeric, " + \
                                   "nucleus_hematoxylin_od_range numeric, " + \
                                   "nucleus_eosin_od_mean numeric, " + \
                                   "nucleus_eosin_od_sum numeric, " + \
                                   "nucleus_eosin_od_std_dev numeric, " + \
                                   "nucleus_eosin_od_max numeric, " + \
                                   "nucleus_eosin_od_min numeric, " + \
                                   "nucleus_eosin_od_range numeric, " + \
                                   "cell_area numeric, " + \
                                   "cell_perimeter numeric, " + \
                                   "cell_circularity numeric, " + \
                                   "cell_max_caliper numeric, " + \
                                   "cell_min_caliper numeric, " + \
                                   "cell_eccentricity numeric, " + \
                                   "cell_hematoxylin_od_mean numeric, " + \
                                   "cell_hematoxylin_od_std_dev numeric, " + \
                                   "cell_hematoxylin_od_max numeric, " + \
                                   "cell_hematoxylin_od_min numeric, " + \
                                   "cell_eosin_od_mean numeric, " + \
                                   "cell_eosin_od_std_dev numeric, " + \
                                   "cell_eosin_od_max numeric, " + \
                                   "cell_eosin_od_min numeric, " + \
                                   "cytoplasm_hematoxylin_od_mean numeric, " + \
                                   "cytoplasm_hematoxylin_od_std_dev numeric, " + \
                                   "cytoplasm_hematoxylin_od_max numeric, " + \
                                   "cytoplasm_hematoxylin_od_min numeric, " + \
                                   "cytoplasm_eosin_od_mean numeric, " + \
                                   "cytoplasm_eosin_od_std_dev numeric, " + \
                                   "cytoplasm_eosin_od_max numeric, " + \
                                   "cytoplasm_eosin_od_min numeric, " + \
                                   "nucleus_cell_area_ratio numeric, " + \
                                   "CONSTRAINT image_parameter_pkey PRIMARY KEY (id) " + \
                                   ");"
    query_create_herohe_data = "CREATE TABLE herohe_data (" + \
                               "id serial NOT NULL," + \
                               "name character varying," + \
                               "image_id integer," + \
                               "her2status integer," + \
                               "mean_nucleus_area double precision," + \
                               "number_of_nucleus_area_small double precision," + \
                               "number_of_nucleus_area_medium double precision," + \
                               "number_of_nucleus_area_large double precision," + \
                               "number_of_nucleus_area_extralarge double precision," + \
                               "mean_nucleus_circularity double precision," + \
                               "mean_nucleus_circularity_small double precision," + \
                               "mean_nucleus_circularity_medium double precision," + \
                               "mean_nucleus_circularity_large double precision," + \
                               "mean_nucleus_circularity_extralarge double precision," + \
                               "mean_nucleus_hematoxylin_od_mean double precision," + \
                               "nucleus_hematoxylin_od_mean_small double precision," + \
                               "nucleus_hematoxylin_od_mean_medium double precision," + \
                               "nucleus_hematoxylin_od_mean_large double precision," + \
                               "nucleus_hematoxylin_od_mean_extralarge double precision," + \
                               "aria_circularity_mean double precision," + \
                               "aria_circularity_density_mean double precision," + \
                               "aria_circularity_mean_small double precision," + \
                               "aria_circularity_density_mean_small double precision," + \
                               "aria_circularity_mean_medium double precision," + \
                               "aria_circularity_density_mean_medium double precision," + \
                               "aria_circularity_mean_large double precision," + \
                               "aria_circularity_density_mean_large double precision," + \
                               "aria_circularity_mean_extralarge double precision," + \
                               "aria_circularity_density_mean_extralarge double precision," + \
                               "number_of_nucleus_circularity_small double precision," + \
                               "number_of_nucleus_circularity_medium double precision," + \
                               "number_of_nucleus_circularity_large double precision," + \
                               "CONSTRAINT herohe_data_pkey PRIMARY KEY (id)" + \
                               ");"

    pgCursor.execute(query_create_images)
    pgCursor.execute(query_create_image_parameter)
    pgCursor.execute(query_create_herohe_data)
    pgConnection.commit()

def importDetectionData():
    for root, dirs, files in os.walk("D:\GoogleDrive\Arbeit\HEROHE_Challenge\ExtractedDataV2", topdown=False):
       for file in files:
          if file.endswith(".txt"):
              if file.endswith("mrxs_TEST.txt"):
                 #print(file.replace(".mrxs", "").replace(".txt", ".mrxs"))
                 print("INSERT INTO images(name, her2status) VALUES ('" + file.replace(".mrxs", "").replace(".txt", ".mrxs") + "', 'Test');")

print("Starting DB creation.")
dropTables()
createTables()

pgConnection.commit()
pgConnection.close()