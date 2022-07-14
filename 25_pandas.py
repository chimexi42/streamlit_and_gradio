import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns

print(pd.__version__)
print(np.__version__)

df = pd.DataFrame({'col_one':[100, 200], 'col_two':[300, 400]})
print(df)

df2 = pd.DataFrame(np.random.rand(4,8))
print(df2)

df3 = pd.DataFrame(np.random.rand(4,8), columns=list("abcdefgh"))
print(df3)

print(df.columns)
df = df.rename({'col_one': 'col one', 'col_two': 'col two'}, axis = 'columns')
print(df)
df.columns =['col_one', 'col_two']
print(df)

df.columns = df.columns.str.replace('_', ' ')
print(df)

print(df.add_prefix('X_'))
print(df.add_suffix('y_'))

drinks = pd.read_csv('drinks.csv')
print(drinks.head())

# Reverse row order
print(drinks.loc[::-1].head())

print(drinks.loc[::-1].reset_index(drop=True).head())
print()


# Reverse columns order
print(drinks.loc[:, ::-1].head())
print()
# Select columns by data type
print(drinks.dtypes)

print(drinks.select_dtypes(include="number").head())

print(drinks.select_dtypes(include="object").head())

print(drinks.select_dtypes(include=["object", "number", "category", "datetime"]).head())

print(drinks.select_dtypes(exclude="number").head())

# convert strings to numbers
df4 = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],
                    'col_two':['4.4', '5.5', '6.6'],
                    'col_three':['7.7','8.8','-']
                    })
print(df4)
print(df4.dtypes)
print(df4.astype({'col_one': 'float', 'col_two':'float'}).dtypes)

print(pd.to_numeric(df4.col_three, errors='coerce'))

print(pd.to_numeric(df4.col_three, errors='coerce').fillna(0))

df4 = df4.apply(pd.to_numeric, errors ='coerce').fillna(0)
print(df4)
print(df4.dtypes)

# Reduce Dataframe size
print(drinks.info(memory_usage='deep'))

cols =['beer_servings', 'country']
small_drinks = pd.read_csv('drinks1.csv', usecols= cols)
print(small_drinks.head())
print(small_drinks.info(memory_usage='deep'))

dtypes = {'continent':'category'}
smaller_drinks = pd.read_csv('drinks2.csv', usecols=['continent', 'wine_servings'], dtype=dtypes)
print(smaller_drinks)
print(smaller_drinks.info(memory_usage='deep'))

# build a dataframe from multiple files

stocks1 = pd.read_csv('stocks.csv')
stocks2 = pd.read_csv('stocks2.csv')
stocks3 = pd.read_csv('stocks3.csv')

stock_files = sorted(glob('stocks*.csv'))
print(stock_files)

stock_df = pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
print(stock_df)

# print(sorted(glob('stocks*.csv')))
#
dfs = (pd.read_csv(file) for file in stock_files)
print(pd.concat(dfs, ignore_index=True))

# Build a dataframe from multiple files column wise

drink_files = sorted(glob('drink*.csv'))
print(drink_files)

print(pd.concat((pd.read_csv(file) for file in drink_files), axis='columns'))

# spliit a dataframe into two random subsets
movies = pd.read_csv('imdb_1000.csv')
print(len(movies))

movies1 = movies.sample(frac=0.75, random_state=1234)
movies2 = movies.drop(movies1.index)
print(movies1)
print(movies2)

print(len(movies1) + len(movies2))

print(movies1.index)
print(movies1.index.sort_values())
print(movies2.index.sort_values())

# filter dataframe by multiple categories
print(movies.columns)
print(movies.genre.unique)

movies_list = movies[(movies.genre == "Action")| (movies.genre == "Drama")| (movies.genre == "Western")].head()
print(movies_list)

print(movies[movies.genre.isin(['Action', 'Drama', 'Western'])])
print(movies[~movies.genre.isin(['Action', 'Drama', 'Western'])])

# filter a dataframe by largest categories
counts = movies.genre.value_counts()
print(counts)
print(counts.nlargest(3))
print(counts.nlargest(3).index)

print(movies[movies.genre.isin(counts.nlargest(3).index)].head())

# Handle missing values
ufo = pd.read_csv('ufo.csv')
print(ufo.head())
print(ufo.isna().sum())
print(ufo.isna().mean())

print(ufo.dropna(axis='columns').head())

print(ufo.dropna(thresh= len(ufo)*0.9, axis ='columns').head())

# split a string into multiple columns

df = pd.DataFrame({"name":['John Arthur Doe', "Jane Ann Smith"],
                   "location":["Los Angeles, CA", "Washington, DC"]})

print(df)
print()

print(df.name.str.split(" "))
print(df.name.str.split(" ", expand =True))
print()

df[["first", "middle", "last"]] = df.name.str.split(" ", expand=True)

print(df)
print()

print(df.location.str.split(",", expand = True))

df["city"] = df.location.str.split(",", expand=True)[0]
print(df)

# Expand a series of list into a dataframe

df = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10,40], [20,50], [30,60]]})
print(df)

df_new = df.col_two.apply(pd.Series)
print(df_new)

print(pd.concat([df, df_new], axis ='columns'))
print()

orders = pd.read_table('chipotle.txt', sep="\t")
print(orders)

item_price_sum = orders[orders.order_id == 2].item_price.sum()
print("item Price: ", item_price_sum)

print()

grouped_by = orders.groupby('order_id').item_price.sum().head()
print()
print(orders.groupby('order_id').item_price.agg(['sum', 'count']).head())
print()
# combine the output of an aggregation with a dataframe

total_price = orders.groupby('order_id').item_price.transform('sum')
print(total_price)
print()

orders['total_price'] = total_price
print(orders.head())

# Select a slice of rows and columns
titanic = pd.read_csv('titanic_test.csv')
print(titanic.head())

print(titanic.describe().loc['min':'max'])
print(titanic[['PassengerId', 'Age']].describe().loc['min':'max'])

print(titanic.describe().loc['min':'max', 'Pclass':'Parch'])

print(titanic.columns)

titanic = sns.load_dataset('titanic')
print(titanic.survived.mean())
print(titanic.survived.count())

print(titanic.groupby('sex').survived.mean())
print()
print(titanic.groupby('sex').survived.count())
print()
print(titanic.groupby(['sex', 'pclass']).survived.count())
print()
print(titanic.groupby(['sex', 'pclass']).survived.mean())
print()
print(titanic.groupby(['sex', 'pclass']).survived.mean().unstack())
print()

# using pivot table
pivoted = titanic.pivot_table(index = 'sex', columns= 'pclass', values = 'survived', aggfunc=['count', 'mean'] )
print(pivoted)
pivoted2 = titanic.pivot_table(index = 'sex', columns= 'pclass', values = 'survived', aggfunc= 'mean' )
print(pivoted2)
pivoted3 = titanic.pivot_table(index = 'sex', columns= 'pclass', values = 'survived', aggfunc= 'count', margins=True )
print(pivoted3)

# converting continous variable into categorical data
print(titanic.age.head())
age_groups = pd.cut(titanic.age, bins = [0, 18, 25, 99], labels=['child', 'young adult', 'adult'])
print(age_groups)

# changing display options
pd.set_option("display.float_format", "{:.2f}".format)
print(titanic.fare.head())

# styling a dataframe
format_dict = {'Date': '{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}

stocks = stocks1.style.format(format_dict)
print(stocks)
