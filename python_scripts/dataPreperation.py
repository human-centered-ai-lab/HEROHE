import os
import psycopg2
from tqdm import tqdm

#################################
# Update Path and DB connection #
#################################
import_folder_path = "D:\GoogleDrive\Arbeit\HEROHE_Challenge\TestExtractedDataV2"
pgConnectString = "host='localhost' port='5432' dbname='heroshetest' user='robert' password='fenris'"


#################################
# Create DB structure           #
#################################

pgConnection=psycopg2.connect(pgConnectString)
pgCursor = pgConnection.cursor()

def dropTables():
    query_drop_images = "DROP TABLE IF EXISTS images;"
    query_drop_image_parameter = "DROP TABLE IF EXISTS image_parameter;"
    query_drop_herohe_data = "DROP TABLE IF EXISTS herohe_data;"
    query_drop_lerndata = "DROP TABLE IF EXISTS lerndata;"
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
    for root, dirs, files in os.walk(import_folder_path, topdown=False):
       for file in tqdm(files):
          if file.endswith(".txt"):
              #print(os.path.join(root, file))
              query_insert_csv = "COPY image_parameter (filename, name, class, parent, roi, centroid_x_um, centroid_y_um, nucleus_area, nucleus_perimeter, nucleus_circularity, nucleus_max_caliper, " + \
                      " nucleus_min_caliper, nucleus_eccentricity, nucleus_hematoxylin_od_mean, nucleus_hematoxylin_od_sum, nucleus_hematoxylin_od_std_dev, nucleus_hematoxylin_od_max, " + \
                      "  nucleus_hematoxylin_od_min, nucleus_hematoxylin_od_range, nucleus_eosin_od_mean, nucleus_eosin_od_sum, nucleus_eosin_od_std_dev, nucleus_eosin_od_max, " + \
                      "       nucleus_eosin_od_min, nucleus_eosin_od_range, cell_area, cell_perimeter, cell_circularity, cell_max_caliper, cell_min_caliper, cell_eccentricity, " + \
                      "           cell_hematoxylin_od_mean, cell_hematoxylin_od_std_dev, cell_hematoxylin_od_max, cell_hematoxylin_od_min, cell_eosin_od_mean, cell_eosin_od_std_dev, " + \
                      "           cell_eosin_od_max, cell_eosin_od_min, cytoplasm_hematoxylin_od_mean, cytoplasm_hematoxylin_od_std_dev, cytoplasm_hematoxylin_od_max, cytoplasm_hematoxylin_od_min, " + \
                      "           cytoplasm_eosin_od_mean, cytoplasm_eosin_od_std_dev, cytoplasm_eosin_od_max, cytoplasm_eosin_od_min, nucleus_cell_area_ratio " + \
                      " ) FROM '" + os.path.join(root, file) + "' DELIMITERS E'\t' CSV HEADER;"
              pgCursor.execute("INSERT INTO images(name) VALUES ('" + file.replace(".txt", "") + "');")
              pgCursor.execute(query_insert_csv)
    pgConnection.commit()

def linkAndUpdateDatabase():
    query_updare_link = "UPDATE image_parameter SET image_id=sub.id FROM ( " + \
                        "SELECT id, name fn FROM images) sub " \
                        "WHERE image_parameter.filename = sub.fn;"
    query_update_cell_count = "UPDATE images SET cell_count = sub.valuecounter FROM (SELECT image_id, COUNT(*) valuecounter FROM image_parameter GROUP BY image_id) sub WHERE sub.image_id = images.id;"
    query_create_tmp_table = "SELECT ip.id, ip.image_id, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity, nucleus_hematoxylin_od_mean, nucleus_cell_area_ratio, " + \
                            "(nucleus_area/nucleus_circularity) ratio1, (nucleus_area/nucleus_circularity/nucleus_hematoxylin_od_mean) ratio2 " + \
                            "INTO lerndata FROM images i " + \
                            "JOIN image_parameter ip ON i.id = ip.image_id;"
    query_insert_images = "INSERT INTO public.herohe_data(name, image_id) (SELECT name, id FROM public.images);"
    pgCursor.execute(query_updare_link)
    pgCursor.execute(query_update_cell_count)
    pgCursor.execute(query_create_tmp_table)
    pgCursor.execute(query_insert_images)
    pgConnection.commit()

def updateParameter():
    querys_update_parameters = []
    querys_update_parameters.append("UPDATE herohe_data SET mean_nucleus_area=sub.upvalue " + \
            "FROM " + \
            "(SELECT image_id, AVG(nucleus_area) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "GROUP BY image_id) sub " + \
            "WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_area_small =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area < 18 " + \
            "GROUP BY image_id) " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_area_medium  =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 18 AND nucleus_area < 54 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_area_large  =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 54 AND nucleus_area <= 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_area_extralarge  =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area > 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET mean_nucleus_circularity =sub.upvalue " + \
            "FROM (SELECT image_id, AVG(nucleus_circularity) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "GROUP BY image_id) sub " + \
            "WHERE sub.image_id = herohe_data.image_id;")
    querys_update_parameters.append("UPDATE herohe_data SET  mean_nucleus_circularity_small  =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_circularity) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area < 18 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  mean_nucleus_circularity_medium   =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_circularity) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 18 AND nucleus_area < 54 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  mean_nucleus_circularity_large   =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_circularity) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 54 AND nucleus_area <= 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  mean_nucleus_circularity_extralarge   =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_circularity) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area > 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET mean_nucleus_hematoxylin_od_mean =sub.upvalue " + \
            "FROM " + \
            "(SELECT image_id, AVG(nucleus_hematoxylin_od_mean) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "GROUP BY image_id) sub " + \
            "WHERE sub.image_id = herohe_data.image_id;")
    querys_update_parameters.append("UPDATE herohe_data SET  nucleus_hematoxylin_od_mean_small   =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_hematoxylin_od_mean) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area < 18 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  nucleus_hematoxylin_od_mean_medium=sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_hematoxylin_od_mean) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 18 AND nucleus_area < 54 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  nucleus_hematoxylin_od_mean_large=sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_hematoxylin_od_mean) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 54 AND nucleus_area <= 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  nucleus_hematoxylin_od_mean_extralarge=sub.upvalue FROM " + \
            "(SELECT image_id, AVG(nucleus_hematoxylin_od_mean) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area > 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET aria_circularity_mean =sub.upvalue " + \
            "FROM " + \
            "(SELECT image_id, AVG(ratio1) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "GROUP BY image_id) sub " + \
            "WHERE sub.image_id = herohe_data.image_id;")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_mean_small =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio1) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area < 18 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_mean_medium =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio1) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 18 AND nucleus_area < 54 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_mean_large =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio1) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 54 AND nucleus_area <= 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_mean_extralarge =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio1) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area > 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET aria_circularity_density_mean  =sub.upvalue " + \
            "FROM " + \
            "(SELECT image_id, AVG(ratio2) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "GROUP BY image_id) sub " + \
            "WHERE sub.image_id = herohe_data.image_id;")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_density_mean_small  =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio2) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area < 18 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_density_mean_medium  =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio2) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 18 AND nucleus_area < 54 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_density_mean_large =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio2) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area >= 54 AND nucleus_area <= 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  aria_circularity_density_mean_extralarge  =sub.upvalue FROM " + \
            "(SELECT image_id, AVG(ratio2) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_area > 82 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_circularity_small  =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_circularity = 1 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_circularity_medium   =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_circularity < 1 AND nucleus_circularity > 0.5 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id; ")
    querys_update_parameters.append("UPDATE herohe_data SET  number_of_nucleus_circularity_large =sub.upvalue FROM " + \
            "(SELECT image_id, (CAST(COUNT(*) AS double precision)/CAST(MAX(cell_count) AS double precision)) upvalue " + \
            "FROM images i " + \
            "JOIN lerndata l ON i.id = l.image_id " + \
            "WHERE nucleus_circularity <= 0.5 " + \
            "GROUP BY image_id)  " + \
            "sub WHERE sub.image_id = herohe_data.image_id;")

    for query in tqdm(querys_update_parameters):
        pgCursor.execute(query)
    pgConnection.commit()

print("Starting DB creation.")
dropTables()
createTables()
importDetectionData()
linkAndUpdateDatabase()
updateParameter()

pgConnection.commit()
pgConnection.close()