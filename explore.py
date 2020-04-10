from __future__ import print_function
from pyspark.sql import SparkSession
import pyspark.sql.functions
from pyspark.sql.functions import col, desc, when

# !spark-submit explore.py
if __name__ == "__main__":

    # Create a SparkSession (Note, the config section is only for Windows!)
    spark = SparkSession.builder.config(
        "spark.sql.warehouse.dir",
        "file:///C:/temp") \
    .appName("LinearRegression") \
    .getOrCreate()

    # NOTE:
    # 1) csv file has headers,
    # 2) csv is in same directory as script
    #    if different location, update load() parameter w/ correct path
    colleges = spark.read.format('csv') \
    .options(header='true', inferSchema='true') \
    .load('College.csv')

    colleges = colleges.withColumnRenamed('_c0', 'name') \
        .withColumnRenamed('Private','is_private') \
        .withColumnRenamed('Apps', 'num_applied') \
        .withColumnRenamed('Accept', 'num_accepted') \
        .withColumnRenamed('Enroll', 'num_enrolled') \
        .withColumnRenamed('Top10perc', 'pct_top10') \
        .withColumnRenamed('Top25perc', 'pct_top25') \
        .withColumnRenamed('F.Undergrad', 'num_ft_undergrad') \
        .withColumnRenamed('P.Undergrad', 'num_pt_undergrad') \
        .withColumnRenamed('Outstate', 'cost_outstate') \
        .withColumnRenamed('Room.Board', 'cost_room_board') \
        .withColumnRenamed('Books', 'cost_books') \
        .withColumnRenamed('Personal', 'cost_personal') \
        .withColumnRenamed('PhD', 'pct_fac_phd') \
        .withColumnRenamed('Terminal', 'pct_fac_term_deg') \
        .withColumnRenamed('S.F.Ratio', 'sf_ratio') \
        .withColumnRenamed('perc.alumni', 'pct_alumni_give') \
        .withColumnRenamed('Expend', 'expense_per_sdnt') \
        .withColumnRenamed('Grad.Rate', 'grad_rate')


    # Convert is_private to 2 columns 'is_private' (1 = yes) and is_public
    colleges = colleges.withColumn('is_private',
    when((col('is_private') == 'Yes'), 1).otherwise(0))

    colleges = colleges.withColumn('is_public',
    when((col('is_private') == 0), 1).otherwise(0))

    colleges.select([c for c in colleges.columns if c in [
        'name',
        'is_public',
        'is_private']]) \
        .show()

    #####################################################################
    # Calculated Columns
    #####################################################################

    # percentage of full time students
    colleges = colleges.withColumn('pct_ft_undergrad',
    col('num_ft_undergrad') / (col('num_ft_undergrad') + col('num_pt_undergrad')))

    # Schools where people are most likely to work
    print('Working Schools (lowest percentage of fulltime students)')
    colleges.select([c for c in colleges.columns if c in [
        'name',
        'num_ft_undergrad',
        'num_pt_undergrad',
        'pct_ft_undergrad']]) \
    .show()

    # acceptance rate
    colleges = colleges.withColumn('pct_accepted',
        col('num_accepted') / col('num_applied'))

    # Most Selective Colleges
    print('Reach Schools (lowest acceptance rates)')
    colleges.select([c for c in colleges.columns if c in [
        'name',
        'num_applied',
        'num_accepted',
        'pct_accepted']]) \
    .sort(col('pct_accepted')) \
    .show()

    # enrollment rate
    colleges = colleges.withColumn('pct_enrolled',
    col('num_enrolled') / col('num_accepted'))

    # Safety Schools
    print('Safety Schools (lowest enrollment rate)')
    colleges.select([c for c in colleges.columns if c in [
        'name',
        'num_accepted',
        'num_enrolled',
        'pct_enrolled']]) \
    .sort(col('pct_enrolled')) \
    .show()


    # total cost
    colleges = colleges.withColumn('total_cost',
    col('cost_outstate') + col('cost_room_board') + col('cost_books') + col('cost_personal'))

    # Most Expensive
    print('Most Expensive')
    colleges.select([c for c in colleges.columns if c in [
        'name',
        'cost_outstate',
        'cost_room_board',
        'cost_books',
        'cost_personal',
        'total_cost']]) \
    .sort(col('total_cost')\
    .desc()) \
    .show()

    # Least Expensive
    print('Least Expensive')
    colleges.select([c for c in colleges.columns if c in [
        'name',
        'cost_outstate',
        'cost_room_board',
        'cost_books',
        'cost_personal',
        'total_cost']]) \
        .sort(col('total_cost')) \
    .show()


    #####################################################################
    # Interesting descriptives (i.e. no causal inference or ml prediction)
    #####################################################################
    # (not sure how to do this w/o dropping into pandas and matplotlib)
    import pandas as pd
    import matplotlib.pyplot as plt

    def plot_scatter(x,y,t,xlab,ylab):
        plt.figure()
        plt.plot(x,y, 'bo')
        plt.title(t)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()


    # Do univerities with a higer sf_ratio have a higher grad_rate?
    df1 = colleges.select([c for c in colleges.columns if c in [
        'sf_ratio',
        'grad_rate']]) \
    .filter(col('grad_rate') <= 100) \
    .toPandas()

    plot_scatter(
        df1.sf_ratio,
        df1.grad_rate,
        'What is the relationship between student \n faculty ratios and graduation rates?',
        'Student Faculty Ratio',
        'Graduation Rate (%)'
        )


    # What is the relationship between prestige (top10% or top25%) and grad_rate?
    df2 = colleges.select([c for c in colleges.columns if c in [
        'pct_top10',
        'grad_rate']]) \
    .filter((col('grad_rate') <= 100) & (col('pct_top10') <= 100)) \
    .toPandas()

    plot_scatter(
        df2.pct_top10,
        df2.grad_rate,
        'What is the relationship between \n prestige (Top 10%) and graduation rates?',
        'Percentage of Incoming Students in Top 10% of H.S. Class',
        'Graduation Rate (%)'
        )

    df3 = colleges.select([c for c in colleges.columns if c in [
        'pct_top25',
        'grad_rate']]) \
    .filter((col('grad_rate') <= 100) & (col('pct_top25') <= 100)) \
    .toPandas()

    plot_scatter(
        df3.pct_top25,
        df3.grad_rate,
        'What is the relationship between \n prestige (Top 25%) and graduation rates?',
        'Percentage of Incoming Students in Top 25% of H.S. Class',
        'Graduation Rate (%)'
        )

    # What is the relationship between prestige and alumni giving?
    df4 = colleges.select([c for c in colleges.columns if c in [
        'pct_top10',
        'pct_alumni_give']]) \
    .filter((col('pct_top10') <= 100) & (col('pct_alumni_give') <= 100)) \
    .toPandas()

    plot_scatter(
        df4.pct_top10,
        df4.pct_alumni_give,
        'What is the relationship between \n prestige (Top 10%) and alumni giving?',
        'Percentage of Incoming Students in Top 10% of H.S. Class',
        'Percentage of Alumni Who Give'
        )

    df5 = colleges.select([c for c in colleges.columns if c in [
        'pct_top25',
        'pct_alumni_give']]) \
    .filter((col('grad_rate') <= 100) & (col('pct_top25') <= 100)) \
    .toPandas()

    plot_scatter(
        df5.pct_top25,
        df5.pct_alumni_give,
        'What is the relationship between \n prestige (Top 25%) and graduation rates?',
        'Percentage of Incoming Students in Top 25% of H.S. Class',
        'Percentage of Alumni Who Give'
        )


    # What is the relationship between alumni giving and graduation rates?
    df6 = colleges.select([c for c in colleges.columns if c in [
        'grad_rate',
        'pct_alumni_give']]) \
    .filter((col('grad_rate') <= 100) & (col('pct_alumni_give') <= 100)) \
    .toPandas()

    plot_scatter(
        df6.pct_alumni_give,
        df6.grad_rate,
        'What is the relationship between \n graduation rates and alumni giving?',
        'Percentage of Alumni Who Give',
        'Graduation Rate (%)'
        )

    # What is the relationship between cost and alumni giving?
    df7 = colleges.select([c for c in colleges.columns if c in [
        'total_cost',
        'pct_alumni_give']]) \
    .filter((col('pct_alumni_give') <= 100)) \
    .toPandas()

    plot_scatter(
        df7.pct_alumni_give,
        df7.total_cost,
        'What is the relationship between \n cost and alumni giving?',
        'Percentage of Alumni Who Give',
        'Graduation Rate (%)'
        )

    # What is the relationship between cost and acceptance rate?
    df8 = colleges.select([c for c in colleges.columns if c in [
        'total_cost',
        'pct_accepted']]) \
    .filter(col('pct_accepted') <= 100) \
    .toPandas()

    plot_scatter(
        df8.pct_accepted,
        df8.total_cost,
        'What is the relationship between \n cost and acceptance rate?',
        'Acceptance Rate (%)',
        'Total Cost ($)'
        )

    # What is the relationship between cost and enrollment rate?
    df9 = colleges.select([c for c in colleges.columns if c in [
        'total_cost',
        'pct_enrolled']]) \
    .filter(col('pct_enrolled') <= 100) \
    .toPandas()

    plot_scatter(
        df9.pct_enrolled,
        df9.total_cost,
        'What is the relationship between \n cost and enrollment rate?',
        'Enrollment Rate (%)',
        'Total Cost ($)'
        )

    # What is the relationship between cost and enrollment rate?
    df10 = colleges.select([c for c in colleges.columns if c in [
        'total_cost',
        'grad_rate']]) \
    .filter(col('grad_rate') <= 100) \
    .toPandas()

    plot_scatter(
        df10.total_cost,
        df10.grad_rate,
        'What is the relationship between \n cost and and graduation rate?',
        'Total Cost ($)',
        'Graduation Rate (%)'
        )
    # Stop the session
    spark.stop()

# !spark-submit explore.py
