import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def combine_product_info(row):
    """Normalize product name"""
    product_name = row['product_name']
    product_name = product_name.replace('woolworths', '').strip()
    if pd.isna(row['weight_or_amount']) or pd.isna(row['weight_or_amount_unit']):
        return product_name.title()
    else:
        return f"{product_name.title()} ({row['weight_or_amount']} {row['weight_or_amount_unit']})"

def preprocess_data(df):
    """
    Preprocess the DataFrame:
    - Combine price_dollar and price_cent into productprice
    - Convert date column to datetime
    - Titleize platform
    - Normalize product name
    """
    df['productprice'] = df['price_dollar'] + df['price_cent'] / 100
    df['date'] = pd.to_datetime(df['date'])
    df['platform'] = df['platform'].str.title()
    df['productname'] = df.apply(combine_product_info, axis=1)
    return df

def filter_products_by_search(df, search_terms):
    """Filter products based on user search terms."""
    if search_terms:
        search_words = search_terms.lower().split()
        for search_word in search_words:
            df = df[df['productname'].str.contains(search_word, case=False, na=False)]
    return df

def generate_price_info_df(df, product_options):
    """Generate DataFrame containing price information for selected products."""
    price_info_df = pd.DataFrame(columns=['Product Name', 'Platform', 'Current Price', 'Lowest Price', 'Lowest Price Date'])

    for product_option in product_options:
        product_df = df[df['productname'] == product_option]
        platforms = product_df['platform'].unique()

        for platform in platforms:
            platform_df = product_df[product_df['platform'] == platform]
            if not platform_df.empty:
                current_price = platform_df.iloc[-1]['productprice']
                last_30_days = platform_df[platform_df['date'] >= platform_df['date'].max() - pd.Timedelta(days=30)]
                if not last_30_days.empty:
                    lowest_price = last_30_days['productprice'].min()
                    lowest_price_date = last_30_days[last_30_days['productprice'] == lowest_price]['date'].max().strftime('%Y-%m-%d')

                    new_row = pd.DataFrame({'Product Name': [product_option],
                                            'Platform': [platform],
                                            'Current Price': [current_price],
                                            'Lowest Price': [lowest_price],
                                            'Lowest Price Date': [lowest_price_date]})
                    price_info_df = pd.concat([price_info_df, new_row], ignore_index=True)
    return price_info_df

def plot_price_trends(df, product_options):
    """Plot price trends for selected products across different platforms."""
    fig, ax = plt.subplots(len(product_options), 1, figsize=(10, 6 * len(product_options)))
    fig.subplots_adjust(hspace=0.5)

    if len(product_options) == 1:
        ax = [ax]

    for i, product_option in enumerate(product_options):
        product_df = df[df['productname'] == product_option]
        platforms = product_df['platform'].unique()

        for platform in platforms:
            platform_df = product_df[product_df['platform'] == platform]
            ax[i].plot(platform_df['date'], platform_df['productprice'], marker='o', label=platform)
            ax[i].set_xlabel('Date')
            ax[i].set_ylabel('Price')
            ax[i].set_title(f'Price Trend for {product_option}')
            ax[i].legend()
            ax[i].tick_params(axis='x', rotation=45)
            ax[i].set_ylim(bottom=0, top=product_df['productprice'].max() * 1.1)

    st.pyplot(fig)

def main():
    st.set_page_config(layout = 'wide')
    st.title("What You Wanna Buy Today? (Vege/Meat/Fish/Egg/Dairy)")

    # Load the dataset
    df = load_data('test.csv')

    # Preprocess the data
    df = preprocess_data(df)

    # User input for product search
    search_terms = st.text_input("Please Enter product name or keywords (multiple words will be required to appear at the same time):")

    # Filter products based on search terms
    df = filter_products_by_search(df, search_terms)

    if not df.empty:
        st.write(f"Found {len(df['productname'].unique())} products matching any of the terms: '{search_terms}'")

        # Display product options with a multi-select box
        product_options = st.multiselect(
            "Select products (you can select multiple):",
            list(df['productname'].unique()),
            default=None
        )

        # Display the table
        st.write("Price Information for Selected Products:")
        price_info_df = generate_price_info_df(df, product_options)
        st.dataframe(price_info_df, use_container_width=True)

        # Generate the plot
        if st.button('Show Price Trends'):
            st.write(f"Displaying price trends for selected products across different platforms.")
            plot_price_trends(df, product_options)
    else:
        st.write("No products found matching your search.")

if __name__ == "__main__":
    main()
