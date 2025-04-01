
# Data Population Script With PostgreSQL

This script loads CSV datasets into a PostgreSQL database using SQLAlchemy and pandas.

## Prerequisites

- Python 3.10+
- [PostgreSQL](https://www.postgresql.org/download/)
- Required Python packages: `sqlalchemy`, `pandas`, `psycopg2-binary`
  - Install them using pip:
    ```bash
    pip install sqlalchemy pandas psycopg2-binary
    ```


  
You can also download PostgreSQL in Terminal:   
   - **macOS**: Use [Homebrew](https://brew.sh/):
     ```bash
     brew install postgresql
     ```
   - **Windows**: Download and install from the [official site](https://www.postgresql.org/download/windows/).
   - **Ubuntu/Linux**:
     ```bash
     sudo apt update
     sudo apt install postgresql postgresql-contrib
     ```
## Setting Up PostgreSQL Locally

You can set up PostgreSQL using either the command-line tool `psql` or the graphical interface `pgAdmin`. Below are instructions for both methods.

### Option 1: Using `psql`

1. **Start PostgreSQL Service**  
   - **macOS**:
     ```bash
     brew services start postgresql
     ```
   - **Linux**:
     ```bash
     sudo service postgresql start
     ```

2. **Access the PostgreSQL Command Line Interface**:
   ```bash
   psql -U postgres
   ```

3. **Create a New User**:
   ```sql
   CREATE USER your_username WITH PASSWORD 'your_password';
   ```

4. **Create a New Database**:
   ```sql
   CREATE DATABASE DSA3101Bank OWNER your_username;
   ```

5. **Grant All Privileges to the User on the Database**:
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE DSA3101Bank TO your_username;
   ```

6. **Exit `psql`**:
   ```sql
   \q
   ```

### Option 2: Using `pgAdmin`

1. **Install pgAdmin**  
   - Download and install pgAdmin from the [official website](https://www.pgadmin.org/download/).

2. **Launch pgAdmin and Connect to Your PostgreSQL Server**:
   - Open pgAdmin and connect to your local PostgreSQL server. The default server is usually named "PostgreSQL" with the default user "postgres".

3. **Create a New User (Login/Group Role)**:
   - In the **Object Explorer**, navigate to **Login/Group Roles**.
   - Right-click and select **Create > Login/Group Role**.
   - In the **General** tab, enter the **Name** (e.g., `your_username`).
   - In the **Definition** tab, set the **Password**.
   - In the **Privileges** tab, set **Can login?** to **Yes** and assign other privileges as needed.
   - Click **Save** to create the user.

4. **Create a New Database**:
   - Right-click on **Databases** in the **Object Explorer** and select **Create > Database**.
   - In the **General** tab, enter the **Database name** (e.g., `DSA3101Bank`).
   - Set the **Owner** to the user you created earlier (`your_username`).
   - Click **Save** to create the database.

## Configuring the Python Script

1. **Update Database Credentials in the Script**:
   - Modify the `db_credential` string in your Python script to reflect your PostgreSQL username and password:
     ```python
     db_credential = 'postgresql://your_username:your_password@localhost:5432'
     ```

2. **Set the Database Name**:
   - Ensure the `db_name` variable matches the name of the database you created:
     ```python
     db_name = '/DSA3101Bank'
     ```

## Project Structure

Ensure your project directory is structured as follows:

```
.
├── data
│   └── raw
│       ├── digital_marketing_campaign_dataset.csv
│       ├── ...
│  
├── database
│   ├── populate_db.py
│   ├── access_database.py
│   └── README.md
│

```

## Running the Script

Execute the Python script to load the dataset into your PostgreSQL database:

```bash
python populate_db.py
```

This will populate the `DSA3101Bank` database with all raw data under `data/raw` folder, creating table accordingly, i.e. `a1_digital_marketing_campaign`.

