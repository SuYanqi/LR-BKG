import matplotlib.pyplot as plt


class PlotUtil:
    @staticmethod
    def show_boxplot(df, title):
        # 如果传进来的是dataframe，这个可以注释掉
        # df = pd.DataFrame(df)

        print(df.describe())

        df.plot.box(title=title)
        plt.grid(linestyle="--", alpha=0.3)
        plt.show()

    @staticmethod
    def show_bar(name_list, num_list):
        # plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
        plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        plt.show()
