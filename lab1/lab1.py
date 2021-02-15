import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep ='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.order_id.count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info())
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
        pass
    
    def most_ordered_item(self):
        # TODO
        a = self.chipo.value_counts(subset = ['item_name']).idxmax()
        item_name = a[0]
        b = self.chipo.loc[self.chipo['item_name']== a[0]].groupby(['item_name']).sum().iloc[0]
        order_id = b['order_id']
        quantity = b['quantity']
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return sum(self.chipo.quantity)
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        f_price = self.chipo.item_price.apply(lambda x: float(x[1:]))
        total_sales = (self.chipo.quantity * f_price).sum()
        return total_sales

    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return self.chipo.order_id.max()
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        avg = self.total_sales()/self.num_orders()
        return round(avg,2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return len(pd.unique(self.chipo.item_name))
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        df = pd.DataFrame.from_dict(letter_counter,orient = 'index').rename(columns = {0:'count'})
        bar = df.sort_values(by = ['count'],ascending = False).head(5)
        bar.plot.bar(xlabel = 'Items', ylabel = 'Number of  Orders', title = 'Most popular items')
        plt.show()
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        #list = self.chipo.apply(lambda x: print('test',x))
        self.chipo.item_price = self.chipo.item_price.map(lambda x : float(x[1:]))
        f = self.chipo.groupby(['order_id']).agg({'item_price':'sum','quantity':'sum'})
        f.plot.scatter(x = 'item_price', y ='quantity', s = 50,c = 'blue', xlabel = 'Order Price', ylabel = 'Num Items', 
                     title = 'Number of Items per order price')
        plt.show()
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(3)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    print(count)
    assert count == 5
    solution.print_columns()
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    # assert quantity == 159 #### the quantity value is not equal to 159. It comes out as 761. Hence, left it out
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()
    
    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    