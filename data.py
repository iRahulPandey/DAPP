import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Set random seed for reproducibility
np.random.seed(42)

# Constants for the dataset
NUM_RECORDS = 1000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Generate random dates
date_range = (END_DATE - START_DATE).days
random_days = np.random.randint(0, date_range, NUM_RECORDS)
# Convert numpy.int64 to Python int explicitly
dates = [START_DATE + timedelta(days=int(day)) for day in sorted(random_days)]

# Product categories and subcategories
categories = {
    'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras'],
    'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
    'Home & Kitchen': ['Appliances', 'Furniture', 'Cookware', 'Decor', 'Bedding'],
    'Books': ['Fiction', 'Non-fiction', 'Educational', 'Comics', 'Biographies'],
    'Sports': ['Equipment', 'Apparel', 'Footwear', 'Accessories', 'Fitness']
}

# Define price ranges by category
price_ranges = {
    'Electronics': (200, 2000),
    'Clothing': (15, 200),
    'Home & Kitchen': (20, 1000),
    'Books': (10, 100),
    'Sports': (15, 500)
}

# Payment methods
payment_methods = ['Credit Card', 'PayPal', 'Apple Pay', 'Google Pay', 'Bank Transfer']

# Cities and states
cities_by_state = {
    'California': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'San Jose'],
    'New York': ['New York City', 'Buffalo', 'Rochester', 'Albany', 'Syracuse'],
    'Texas': ['Houston', 'Austin', 'Dallas', 'San Antonio', 'Fort Worth'],
    'Florida': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Tallahassee'],
    'Illinois': ['Chicago', 'Springfield', 'Naperville', 'Peoria', 'Rockford']
}

# Generate data
data = []
order_id = 10000

for i in range(NUM_RECORDS):
    date = dates[i]
    
    # Add some seasonality to the data
    month = date.month
    if 10 <= month <= 12:  # Holiday season
        category_weights = [0.4, 0.2, 0.2, 0.1, 0.1]  # Higher weight to electronics
    elif 5 <= month <= 8:  # Summer
        category_weights = [0.2, 0.25, 0.15, 0.1, 0.3]  # Higher weight to sports
    else:
        category_weights = [0.25, 0.2, 0.25, 0.15, 0.15]  # Balanced
    
    category = np.random.choice(list(categories.keys()), p=category_weights)
    subcategory = random.choice(categories[category])
    
    # Generate a product name
    product = f"{subcategory} Product {random.randint(1, 100)}"
    
    # Price with some variation
    min_price, max_price = price_ranges[category]
    price = round(random.uniform(min_price, max_price), 2)
    
    # Quantity - higher priced items tend to have lower quantities
    if price > 500:
        quantity = random.choices([1, 2], weights=[0.8, 0.2])[0]
    elif price > 100:
        quantity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    else:
        quantity = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
    
    total_amount = round(price * quantity, 2)
    
    # Customer info
    customer_id = random.randint(1000, 9999)
    customer_age = random.randint(18, 75)
    customer_gender = random.choice(['Male', 'Female', 'Non-binary'])
    
    state = random.choice(list(cities_by_state.keys()))
    city = random.choice(cities_by_state[state])
    
    payment_method = random.choice(payment_methods)
    
    # Rating
    rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.3, 0.35])[0]
    
    # Discount
    discount_pct = random.choices([0, 5, 10, 15, 20], weights=[0.6, 0.2, 0.1, 0.07, 0.03])[0]
    discount_amount = round(total_amount * (discount_pct / 100), 2)
    final_amount = round(total_amount - discount_amount, 2)
    
    data.append({
        'OrderID': order_id,
        'Date': date.strftime('%Y-%m-%d'),
        'CustomerID': customer_id,
        'Product': product,
        'Category': category,
        'Subcategory': subcategory,
        'Price': price,
        'Quantity': quantity,
        'TotalAmount': total_amount,
        'DiscountPercent': discount_pct,
        'DiscountAmount': discount_amount,
        'FinalAmount': final_amount,
        'PaymentMethod': payment_method,
        'CustomerAge': customer_age,
        'CustomerGender': customer_gender,
        'City': city,
        'State': state,
        'Rating': rating
    })
    
    order_id += 1

# Create the DataFrame
df = pd.DataFrame(data)

# Add some repeat customers to make the data more realistic
customer_frequencies = np.random.exponential(scale=2, size=NUM_RECORDS)
customer_frequencies = np.clip(customer_frequencies, 1, 10).astype(int)

for i, freq in enumerate(customer_frequencies):
    if freq > 1 and i < NUM_RECORDS - freq:
        customer_id = df.iloc[i]['CustomerID']
        for j in range(1, freq):
            if i + j < NUM_RECORDS:
                df.at[i + j, 'CustomerID'] = customer_id

# Save to CSV
df.to_csv('data/online_retail_sales.csv', index=False)

print(f"Dataset created with {len(df)} records")
print(f"File saved to data/online_retail_sales.csv")