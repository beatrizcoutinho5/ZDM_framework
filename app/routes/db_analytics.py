import csv
import os
import time
import psycopg2

from datetime import datetime

# Database connection
DB_HOST = 'db.fe.up.pt'
DB_PORT = '5432'
DB_NAME = 'xxx'
DB_SCHEMA = 'zdm_framework'
DB_USER = 'xxx'
DB_PASSWORD = 'xxx'

def db_save_sample(processed_sample, recording_date, prediction):
    try:
        elapsed_time = time.time()

        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Order of the features
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

        # Order the processed sample
        ordered_processed_sample = {key: processed_sample[key] for key in order if key in processed_sample}

        # SQL query
        columns = ['"Recording Date"', '"Defect Prediction"'] + [f'"{col}"' for col in ordered_processed_sample.keys()]
        placeholders = ', '.join(['%s'] * (len(ordered_processed_sample) + 2))
        query = f"INSERT INTO zdm_framework.ProductionData ({', '.join(columns)}) VALUES ({placeholders})"

        # Values from the processed sample
        values = [recording_date, prediction] + list(ordered_processed_sample.values())

        cursor.execute(query, values)

        conn.commit()
        cursor.close()
        conn.close()

        elapsed_time = time.time() - elapsed_time
        print(f"\nTotal Elapsed Time to save sample: {elapsed_time}")

        print(f"\nSaved sample data to database!")

    except (Exception, psycopg2.DatabaseError) as error:
        return

def db_get_defects_number(start_time=None, end_time=None):

    elapsed_time = time.time()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Get the number of defected sample from the selected time period
    defects_number_query = ("SELECT COUNT(\"Defect Prediction\") "
                            "FROM zdm_framework.ProductionData "
                            f"WHERE \"Defect Prediction\" = '1' "
                            f"AND \"Recording Date\" BETWEEN '{start_time}' AND '{end_time}'")

    cursor.execute(defects_number_query)
    defects_number_result = cursor.fetchone()

    # Get the number of samples from the selected time period
    produced_panels_query = ("SELECT COUNT(\"Defect Prediction\") "
                             "FROM zdm_framework.ProductionData "
                             f"WHERE \"Recording Date\" BETWEEN '{start_time}' AND '{end_time}'")

    cursor.execute(produced_panels_query)
    produced_panels_result = cursor.fetchone()


    # Calculate the percentage of defects
    if produced_panels_result[0] != 0:

        percentage_defect = (defects_number_result[0] / produced_panels_result[0])*100
        percentage_defect = round(percentage_defect, 1)

    else:
        percentage_defect = 0


    # Format dates
    datetime_obj_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    date_part_start = datetime_obj_start.strftime("%Y-%m-%d")

    datetime_obj_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    date_part_end = datetime_obj_end.strftime("%Y-%m-%d")

    # Returns the count defects for each day within the time range
    defects_number_per_day_query = ("SELECT TO_DATE(\"Recording Date\", 'YYYY-MM-DD') AS defect_date, COUNT(\"Defect Prediction\") AS defect_count "
                                    "FROM zdm_framework.ProductionData "
                                    f"WHERE \"Defect Prediction\" = '1' "
                                    f"AND TO_DATE(\"Recording Date\", 'YYYY-MM-DD') BETWEEN '{date_part_start}' AND '{date_part_end}' "
                                    "GROUP BY TO_DATE(\"Recording Date\", 'YYYY-MM-DD') "
                                    "ORDER BY TO_DATE(\"Recording Date\", 'YYYY-MM-DD')")

    cursor.execute(defects_number_per_day_query)
    defects_number_per_day_results = cursor.fetchall()

    defects_number_per_day_results = [(date.strftime('%Y-%m-%d'), count) for date, count in defects_number_per_day_results]

    cursor.close()
    conn.close()

    elapsed_time = time.time() - elapsed_time
    print(f"\nTotal Elapsed Time to get stats: {elapsed_time}")

    return defects_number_result, produced_panels_result, percentage_defect, defects_number_per_day_results


# Get the historic data between the selected dates, but only 3 samples and with a few features for UI display
def db_get_historic_data(start_time=None, end_time=None):

    elapsed_time = time.time()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Format dates
    datetime_obj_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    date_part_start = datetime_obj_start.strftime("%Y-%m-%d")

    datetime_obj_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    date_part_end = datetime_obj_end.strftime("%Y-%m-%d")

    # Get the historic data between the selected dates
    historic_data_query = ( "SELECT \"Recording Date\", \"Defect Prediction\", \"Upper Plate Temperature\", "
                            "\"Lower Plate Temperature\", \"Thermal Cycle Time\", \"Width\", \"Length\", \"Pressure\" "
                            "FROM zdm_framework.ProductionData "
                            f"WHERE TO_DATE(\"Recording Date\", 'YYYY-MM-DD') BETWEEN '{date_part_start}' AND '{date_part_end}' "
                            "GROUP BY \"Recording Date\", \"Defect Prediction\", \"Upper Plate Temperature\", "
                            "\"Lower Plate Temperature\", \"Thermal Cycle Time\", \"Width\", \"Length\", \"Pressure\" "
                            "ORDER BY TO_DATE(\"Recording Date\", 'YYYY-MM-DD') DESC "
                            "LIMIT 3")

    cursor.execute(historic_data_query)
    historic_data = cursor.fetchall()

    cursor.close()
    conn.close()

    elapsed_time = time.time() - elapsed_time
    print(f"\nTotal Elapsed Time to get historical samples: {elapsed_time}")

    return historic_data

# Get the historic data between the selected dates with all sample and features for download
def db_download_historic_data(start_time=None, end_time=None):

    elapsed_time = time.time()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Format dates
    datetime_obj_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    date_part_start = datetime_obj_start.strftime("%Y-%m-%d")

    datetime_obj_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    date_part_end = datetime_obj_end.strftime("%Y-%m-%d")

    # Get the historic data between the selected dates
    historic_data_query = (
        "SELECT * "
        "FROM zdm_framework.ProductionData "
        f"WHERE TO_DATE(\"Recording Date\", 'YYYY-MM-DD') BETWEEN '{date_part_start}' AND '{date_part_end}' "
        "ORDER BY TO_DATE(\"Recording Date\", 'YYYY-MM-DD') ASC "
    )

    cursor.execute(historic_data_query)
    historic_data = cursor.fetchall()

    column_names = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/downloads')
    csv_file_path = os.path.join(static_folder, 'data.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(column_names)

        # Write data
        csv_writer.writerows(historic_data)

        elapsed_time = time.time() - elapsed_time
        print(f"\nTotal Elapsed Time to get historical samples (download): {elapsed_time}")

    return 1





