"""
Creates a SQLite test database populated with sample data
for execution-based evaluation of generated SQL queries.
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta

os.makedirs("data/processed", exist_ok=True)

DB_PATH = "data/processed/finance.db"

# Remove existing database
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create tables
cursor.executescript("""
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        created_at DATE
    );

    CREATE TABLE accounts (
        account_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        account_type TEXT,
        balance DECIMAL(10, 2),
        currency TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE expenses (
        expense_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        category TEXT,
        amount DECIMAL(10, 2),
        date DATE,
        description TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE income (
        income_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        source TEXT,
        amount DECIMAL(10, 2),
        date DATE,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE budgets (
        budget_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        category TEXT,
        monthly_limit DECIMAL(10, 2),
        start_date DATE,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE transactions (
        txn_id INTEGER PRIMARY KEY,
        account_id INTEGER,
        txn_type TEXT,
        amount DECIMAL(10, 2),
        date DATE,
        merchant TEXT,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    );
""")

# Sample data
users_data = [
    (1, "Fadhil", "fadhil@email.com", "2023-01-15"),
    (2, "Rahul", "rahul@email.com", "2023-02-20"),
    (3, "Aisha", "aisha@email.com", "2023-03-10"),
    (4, "Sarah", "sarah@email.com", "2023-04-05"),
    (5, "Michael", "michael@email.com", "2023-05-12"),
    (6, "Emma", "emma@email.com", "2023-06-18"),
    (7, "James", "james@email.com", "2023-07-22"),
    (8, "Priya", "priya@email.com", "2023-08-30"),
]

cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users_data)

# Generate accounts
account_types = ["Savings", "Checking", "Investment", "Credit"]
currencies = ["USD", "EUR", "GBP", "INR"]
account_id = 1
accounts_data = []

for user_id in range(1, 9):
    num_accounts = random.randint(1, 3)
    for _ in range(num_accounts):
        acc_type = random.choice(account_types)
        balance = round(random.uniform(100, 50000), 2)
        currency = random.choice(currencies)
        accounts_data.append((account_id, user_id, acc_type, balance, currency))
        account_id += 1

cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", accounts_data)

# Generate expenses
categories = ["Food", "Travel", "Rent", "Utilities", "Entertainment", "Healthcare", "Shopping", "Transportation"]
expense_id = 1
expenses_data = []

for user_id in range(1, 9):
    num_expenses = random.randint(20, 50)
    for _ in range(num_expenses):
        category = random.choice(categories)
        amount = round(random.uniform(5, 500), 2)
        days_ago = random.randint(0, 180)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        description = f"{category} expense"
        expenses_data.append((expense_id, user_id, category, amount, date, description))
        expense_id += 1

cursor.executemany("INSERT INTO expenses VALUES (?, ?, ?, ?, ?, ?)", expenses_data)

# Generate income
sources = ["Salary", "Freelance", "Dividends", "Rental", "Bonus"]
income_id = 1
income_data = []

for user_id in range(1, 9):
    num_income = random.randint(5, 15)
    for _ in range(num_income):
        source = random.choice(sources)
        amount = round(random.uniform(500, 10000), 2)
        days_ago = random.randint(0, 180)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        income_data.append((income_id, user_id, source, amount, date))
        income_id += 1

cursor.executemany("INSERT INTO income VALUES (?, ?, ?, ?, ?)", income_data)

# Generate budgets
budget_id = 1
budgets_data = []

for user_id in range(1, 9):
    for category in random.sample(categories, k=random.randint(3, 6)):
        monthly_limit = round(random.uniform(100, 1000), 2)
        start_date = "2024-01-01"
        budgets_data.append((budget_id, user_id, category, monthly_limit, start_date))
        budget_id += 1

cursor.executemany("INSERT INTO budgets VALUES (?, ?, ?, ?, ?)", budgets_data)

# Generate transactions
merchants = ["Amazon", "Walmart", "Starbucks", "Netflix", "Uber", "Shell", "Target", "Costco"]
txn_types = ["purchase", "refund", "transfer", "withdrawal", "deposit"]
txn_id = 1
transactions_data = []

for acc_id, user_id, _, _, _ in accounts_data:
    num_txns = random.randint(10, 30)
    for _ in range(num_txns):
        txn_type = random.choice(txn_types)
        amount = round(random.uniform(5, 500), 2)
        days_ago = random.randint(0, 90)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        merchant = random.choice(merchants)
        transactions_data.append((txn_id, acc_id, txn_type, amount, date, merchant))
        txn_id += 1

cursor.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)", transactions_data)

conn.commit()
conn.close()

print(f"Test database created at {DB_PATH}")
print(f"  - {len(users_data)} users")
print(f"  - {len(accounts_data)} accounts")
print(f"  - {len(expenses_data)} expenses")
print(f"  - {len(income_data)} income records")
print(f"  - {len(budgets_data)} budgets")
print(f"  - {len(transactions_data)} transactions")
