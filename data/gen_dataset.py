import json
import os
import random
import yaml

os.makedirs("data/processed", exist_ok=True)

# Load schema from config
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

SCHEMA = config["schema_simple"]

names = ["Fadhil", "Rahul", "Aisha", "Sarah", "Michael", "Emma", "James", "Priya"]
categories = ["Food", "Travel", "Rent", "Utilities", "Entertainment", "Healthcare", "Shopping", "Transportation"]
account_types = ["Savings", "Checking", "Investment", "Credit"]
income_sources = ["Salary", "Freelance", "Dividends", "Rental", "Bonus"]
merchants = ["Amazon", "Walmart", "Starbucks", "Netflix", "Uber", "Shell", "Target", "Costco"]
currencies = ["USD", "EUR", "GBP", "INR"]

data = []

for _ in range(15):
    name = random.choice(names)
    category = random.choice(categories)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: How much did {name} spend on {category} last month?\nSchema: {SCHEMA}",
        "output": f"SELECT SUM(e.amount) FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}' AND e.category = '{category}' AND e.date >= DATE('now', '-1 month');"
    })

for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What is {name}'s total income this year?\nSchema: {SCHEMA}",
        "output": f"SELECT SUM(i.amount) FROM income i JOIN users u ON i.user_id = u.user_id WHERE u.name = '{name}' AND i.date >= DATE('now', 'start of year');"
    })

# count queries
for _ in range(10):
    category = random.choice(categories)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: How many transactions were made in the {category} category?\nSchema: {SCHEMA}",
        "output": f"SELECT COUNT(*) FROM expenses WHERE category = '{category}';"
    })

for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: How many accounts does {name} have?\nSchema: {SCHEMA}",
        "output": f"SELECT COUNT(*) FROM accounts a JOIN users u ON a.user_id = u.user_id WHERE u.name = '{name}';"
    })

# average/ min /max
for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What is the average expense amount for {name}?\nSchema: {SCHEMA}",
        "output": f"SELECT AVG(e.amount) FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}';"
    })

for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What was {name}'s largest expense?\nSchema: {SCHEMA}",
        "output": f"SELECT MAX(e.amount) FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}';"
    })

#group by queries
for _ in range(15):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Show {name}'s total spending by category.\nSchema: {SCHEMA}",
        "output": f"SELECT e.category, SUM(e.amount) AS total FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}' GROUP BY e.category;"
    })

for _ in range(10):
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What is the total spending per user?\nSchema: {SCHEMA}",
        "output": "SELECT u.name, SUM(e.amount) AS total FROM expenses e JOIN users u ON e.user_id = u.user_id GROUP BY u.user_id, u.name;"
    })

for _ in range(10):
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Show monthly income totals for each user.\nSchema: {SCHEMA}",
        "output": "SELECT u.name, strftime('%Y-%m', i.date) AS month, SUM(i.amount) AS total FROM income i JOIN users u ON i.user_id = u.user_id GROUP BY u.user_id, u.name, strftime('%Y-%m', i.date);"
    })

#having queries
for _ in range(10):
    threshold = random.choice([100, 500, 1000, 2000])
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Which categories have total spending over ${threshold}?\nSchema: {SCHEMA}",
        "output": f"SELECT category, SUM(amount) AS total FROM expenses GROUP BY category HAVING SUM(amount) > {threshold};"
    })

for _ in range(10):
    count = random.choice([5, 10, 20])
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Which users have made more than {count} transactions?\nSchema: {SCHEMA}",
        "output": f"SELECT u.name, COUNT(*) AS txn_count FROM expenses e JOIN users u ON e.user_id = u.user_id GROUP BY u.user_id, u.name HAVING COUNT(*) > {count};"
    })

#multitable joins
for _ in range(15):
    name = random.choice(names)
    acc_type = random.choice(account_types)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Show all transactions for {name}'s {acc_type} account.\nSchema: {SCHEMA}",
        "output": f"SELECT t.* FROM transactions t JOIN accounts a ON t.account_id = a.account_id JOIN users u ON a.user_id = u.user_id WHERE u.name = '{name}' AND a.account_type = '{acc_type}';"
    })

for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What is {name}'s net worth (total balance across all accounts)?\nSchema: {SCHEMA}",
        "output": f"SELECT SUM(a.balance) AS net_worth FROM accounts a JOIN users u ON a.user_id = u.user_id WHERE u.name = '{name}';"
    })

# subqueries
for _ in range(10):
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Who spent the most money overall?\nSchema: {SCHEMA}",
        "output": "SELECT u.name, SUM(e.amount) AS total FROM expenses e JOIN users u ON e.user_id = u.user_id GROUP BY u.user_id, u.name ORDER BY total DESC LIMIT 1;"
    })

for _ in range(10):
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Find users who spent more than the average expense amount.\nSchema: {SCHEMA}",
        "output": "SELECT DISTINCT u.name FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE e.amount > (SELECT AVG(amount) FROM expenses);"
    })

for _ in range(10):
    category = random.choice(categories)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Which users have never spent on {category}?\nSchema: {SCHEMA}",
        "output": f"SELECT u.name FROM users u WHERE u.user_id NOT IN (SELECT DISTINCT user_id FROM expenses WHERE category = '{category}');"
    })

# data filtering
for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Show {name}'s expenses from the last 7 days.\nSchema: {SCHEMA}",
        "output": f"SELECT * FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}' AND e.date >= DATE('now', '-7 days');"
    })

for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What is {name}'s total spending this quarter?\nSchema: {SCHEMA}",
        "output": f"SELECT SUM(e.amount) FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}' AND e.date >= DATE('now', 'start of month', '-2 months');"
    })

# order by queries
for _ in range(10):
    name = random.choice(names)
    n = random.choice([3, 5, 10])
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: What are {name}'s top {n} largest expenses?\nSchema: {SCHEMA}",
        "output": f"SELECT * FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}' ORDER BY e.amount DESC LIMIT {n};"
    })

for _ in range(10):
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Show the 5 most recent transactions.\nSchema: {SCHEMA}",
        "output": "SELECT * FROM transactions ORDER BY date DESC LIMIT 5;"
    })

# comparison of budget
for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Has {name} exceeded any budget this month?\nSchema: {SCHEMA}",
        "output": f"SELECT b.category, b.monthly_limit, SUM(e.amount) AS spent FROM budgets b JOIN users u ON b.user_id = u.user_id JOIN expenses e ON b.user_id = e.user_id AND b.category = e.category WHERE u.name = '{name}' AND e.date >= DATE('now', 'start of month') GROUP BY b.category, b.monthly_limit HAVING SUM(e.amount) > b.monthly_limit;"
    })

for _ in range(10):
    name = random.choice(names)
    category = random.choice(categories)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: How much budget does {name} have left for {category}?\nSchema: {SCHEMA}",
        "output": f"SELECT b.monthly_limit - COALESCE(SUM(e.amount), 0) AS remaining FROM budgets b JOIN users u ON b.user_id = u.user_id LEFT JOIN expenses e ON b.user_id = e.user_id AND b.category = e.category AND e.date >= DATE('now', 'start of month') WHERE u.name = '{name}' AND b.category = '{category}' GROUP BY b.monthly_limit;"
    })

# case statements
for _ in range(10):
    name = random.choice(names)
    data.append({
        "instruction": "Write a SQL query for the following question",
        "input": f"Question: Categorize {name}'s expenses as 'Small' (under $50), 'Medium' ($50-$200), or 'Large' (over $200).\nSchema: {SCHEMA}",
        "output": f"SELECT e.*, CASE WHEN e.amount < 50 THEN 'Small' WHEN e.amount BETWEEN 50 AND 200 THEN 'Medium' ELSE 'Large' END AS size_category FROM expenses e JOIN users u ON e.user_id = u.user_id WHERE u.name = '{name}';"
    })

# Shuffle and save
random.shuffle(data)

with open("data/processed/finance_sql.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Dataset with {len(data)} diverse samples created.")