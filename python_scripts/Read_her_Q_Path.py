import os
import psycopg2

pgConnectString = "host='127.0.0.1' port='5432' dbname='postgres' user='postgres' password='postgres'"
pgConnection= psycopg2.connect(pgConnectString)
pgCursor = pgConnection.cursor()

# query = "COPY images (name, her2status) FROM '/home/simon/PycharmProjects/robert_sql/HEROHE_HER2_STATUS.csv' DELIMITERS ';' CSV HEADER;  \
#             Update images SET name = name || '.mrxs'; \
#     UPDATE image_parameter SET image_id = images.id FROM images WHERE images.name = image_parameter.filename; "

for root, dirs, files in os.walk("/home/simon/PycharmProjects/robert_sql/extracts", topdown=False):
    print(files)
    for file in files:
        if file.endswith(".txt") and not "TEST" in file:
            print(os.path.join(root, file))
            query = "COPY image_parameter (filename, name, class, parent, roi, centroid_x_um, centroid_y_um, nucleus_area, nucleus_perimeter, nucleus_circularity, nucleus_max_caliper, " + \
                    " nucleus_min_caliper, nucleus_eccentricity, nucleus_hematoxylin_od_mean, nucleus_hematoxylin_od_sum, nucleus_hematoxylin_od_std_dev, nucleus_hematoxylin_od_max, " + \
                    "  nucleus_hematoxylin_od_min, nucleus_hematoxylin_od_range, nucleus_eosin_od_mean, nucleus_eosin_od_sum, nucleus_eosin_od_std_dev, nucleus_eosin_od_max, " + \
                    "       nucleus_eosin_od_min, nucleus_eosin_od_range, cell_area, cell_perimeter, cell_circularity, cell_max_caliper, cell_min_caliper, cell_eccentricity, " + \
                    "           cell_hematoxylin_od_mean, cell_hematoxylin_od_std_dev, cell_hematoxylin_od_max, cell_hematoxylin_od_min, cell_eosin_od_mean, cell_eosin_od_std_dev, " + \
                    "           cell_eosin_od_max, cell_eosin_od_min, cytoplasm_hematoxylin_od_mean, cytoplasm_hematoxylin_od_std_dev, cytoplasm_hematoxylin_od_max, cytoplasm_hematoxylin_od_min, " + \
                    "           cytoplasm_eosin_od_mean, cytoplasm_eosin_od_std_dev, cytoplasm_eosin_od_max, cytoplasm_eosin_od_min, nucleus_cell_area_ratio " + \
                    " ) FROM '" + os.path.join(root, file) + "' DELIMITERS E'\t' CSV HEADER;"

            pgCursor.execute(query)

pgConnection.commit()
pgConnection.close()


