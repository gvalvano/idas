"""
Utilities for sqlite database data
"""
#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# TODO: this is an incomplete code
# written for the specific case of a 2 columns table (row_id, values)

import sqlite3


def create_db(db_name, table_name, column_names):
    """ Create sqlite database with one table and two columns. 
    :param db_name: name of the db
    :param table_name: name of the table to create
    :param column_names: list of names for the columns
    :return: 
    """
    # db creation:
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table
    c.execute("CREATE TABLE {0} ({1}, {2})".format(table_name, column_names[0], column_names[1]))

    # Save (commit) the changes and close the connection
    conn.commit()
    conn.close()


def insert_values(db_name, table_name, values):
    """ Inset a row of data in the database table 
    :param db_name: name of the tb
    :param table_name: name of the table to fill with values
    :param values: values of the row (2 values)
    :return: 
    """

    # db connection:
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Insert a row of data
    c.execute("INSERT INTO {0} VALUES (?,?)", (table_name, values))
    conn.commit()

    conn.close()
