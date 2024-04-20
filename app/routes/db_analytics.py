import psycopg2
from datetime import datetime

from __main__ import app

# Database connection
DB_HOST = 'db.fe.up.pt'
DB_PORT = '5432'
DB_NAME = 'sie2338'
DB_SCHEMA = 'zdm_framework'
DB_USER = 'sie2338'
DB_PASSWORD = 'logan123'

def db_save_sample(processed_sample, recording_date, prediction):

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # order of keys
    order = ["Production Line", "Production Order Code", "Production Order Opening", "Length", "Width", "Thickness",
             "Lot Size",
             "Cycle Time", "Mechanical Cycle Time", "Thermal Cycle Time", "Control Panel with Micro Stop",
             "Control Panel Delay Time", "Sandwich Preparation Time", "Carriage Time", "Lower Plate Temperature",
             "Upper Plate Temperature",
             "Pressure", "Roller Start Time", "Liston 1 Speed", "Liston 2 Speed", "Floor 1", "Floor 2",
             "Bridge Platform", "Floor 1 Blow Time",
             "Floor 2 Blow Time", "Centering Table", "Conveyor Belt Speed Station 1", "Quality Inspection Cycle",
             "Conveyor Belt Speed Station 2",
             "Transverse Saw Cycle", "Right Jaw Discharge", "Left Jaw Discharge", "Simultaneous Jaw Discharge",
             "Carriage Speed", "Take-off Path",
             "Stacking Cycle", "Lowering Time", "Take-off Time", "High Pressure Input Time",
             "Press Input Table Speed", "Scraping Cycle", "Paper RC",
             "Paper VC", "Paper Shelf Life", "GFFTT_cat", "Finishing Top_cat", "Reference Top_cat"]

    # order the processed sample
    ordered_processed_sample = {key: processed_sample[key] for key in order if key in processed_sample}

    # SQL query
    columns = ', '.join(['"Recording Date"' if recording_date else '',
                         '"Defect Prediction"' if prediction else ''] +
                        [f'"{col}"' for col in ordered_processed_sample.keys()])
    placeholders = ', '.join(['%s'] * (len(ordered_processed_sample) + 2))
    query = f"INSERT INTO zdm_framework.ProductionData ({columns}) VALUES ({placeholders})"

    # values from the processed sample
    values = list(ordered_processed_sample.values())
    values.insert(0, recording_date)
    values.insert(1, prediction)

    cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nSaved sample data to database!")


def db_get_avg_feature_values():

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    query_lpt = ("SELECT AVG(\"Lower Plate Temperature\") "
                 "FROM zdm_framework.ProductionData "
                 "WHERE \"Defect Prediction\" = '1'")

    query_upt = ("SELECT AVG(\"Upper Plate Temperature\") "
                 "FROM zdm_framework.ProductionData "
                 "WHERE \"Defect Prediction\" = '1'")

    query_length = ("SELECT AVG(\"Length\") "
                    "FROM zdm_framework.ProductionData "
                    "WHERE \"Defect Prediction\" = '1'")

    query_width = ("SELECT AVG(\"Width\") "
                   "FROM zdm_framework.ProductionData "
                   "WHERE \"Defect Prediction\" = '1'")

    query_thickness = ("SELECT AVG(\"Thickness\") "
                       "FROM zdm_framework.ProductionData "
                       "WHERE \"Defect Prediction\" = '1'")

    query_pressure = ("SELECT AVG(\"Pressure\") "
                      "FROM zdm_framework.ProductionData "
                      "WHERE \"Defect Prediction\" = '1'")

    query_tct = ("SELECT AVG(\"Thermal Cycle Time\") "
                 "FROM zdm_framework.ProductionData "
                 "WHERE \"Defect Prediction\" = '1'")

    cursor.execute(query_lpt)
    lpt_result = cursor.fetchone()[0]

    cursor.execute(query_upt)
    upt_result = cursor.fetchone()[0]

    cursor.execute(query_length)
    length_result = cursor.fetchone()[0]

    cursor.execute(query_width)
    width_result = cursor.fetchone()[0]

    cursor.execute(query_thickness)
    thickness_result = cursor.fetchone()[0]

    cursor.execute(query_pressure)
    pressure_result = cursor.fetchone()[0]

    cursor.execute(query_tct)
    tct_result = cursor.fetchone()[0]

    print("\nFeatures avg values:")
    print(f"Lower Plate Temperature: {lpt_result}")
    print(f"Upper Plate Temperature: {upt_result}")
    print(f"Length: {length_result}")
    print(f"Width: {width_result}")
    print(f"Thickness: {thickness_result}")
    print(f"Pressure: {pressure_result}")
    print(f"Thermal Cycle Time: {tct_result}")

    cursor.close()
    conn.close()

    return lpt_result, upt_result, length_result, width_result, thickness_result, pressure_result, tct_result



def db_get_defects_number(start_time=None, end_time=None):

    print("entrei")

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    defects_number_query = None

    # no date interval provided, return total defect count
    if start_time is None and end_time is None:

        defects_number_query = ("SELECT COUNT(\"Defect Prediction\") "
                                "FROM zdm_framework.ProductionData "
                                "WHERE \"Defect Prediction\" = '1'")
        cursor.execute(defects_number_query)
        result = cursor.fetchone()
        print(f"Number of defects: {result[0]}")

    # only a start time provided but not as end
    # make the end time today
    elif start_time is not None and end_time is None:

        end_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        defects_number_query = ("SELECT COUNT(\"Defect Prediction\") "
                                "FROM zdm_framework.ProductionData "
                                f"WHERE \"Defect Prediction\" = '1' "
                                f"AND \"Recording Date\" BETWEEN '{start_time}' AND '{end_time}'")
        cursor.execute(defects_number_query)
        result = cursor.fetchone()
        print(f"Number of defects: {result[0]}")

    # provides both the start and end time
    elif start_time is not None and end_time is not None:

        defects_number_query = ("SELECT COUNT(\"Defect Prediction\") "
                                "FROM zdm_framework.ProductionData "
                                f"WHERE \"Defect Prediction\" = '1' "
                                f"AND \"Recording Date\" BETWEEN '{start_time}' AND '{end_time}'")
        cursor.execute(defects_number_query)
        result = cursor.fetchone()
        print(f"Number of defects: {result[0]}")

    cursor.close()
    conn.close()

    return defects_number_query
