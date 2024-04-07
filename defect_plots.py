import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel(r'data\split_train_test_data\with_delta_values\defect_score_optimization\defect_score_analysis.xlsx')

defect_code_0_scores = df[df['Defect Code'] == 0]['Defect Score']
defect_code_1_scores = df[df['Defect Code'] == 1]['Defect Score']

plt.figure(figsize=(12, 6))
bp = plt.boxplot([defect_code_0_scores, defect_code_1_scores], labels=['Defect Code 0', 'Defect Code 1'], vert=False, showfliers=True, showmeans=True, meanline=True)

plt.title('Defect Score Distribution for Each Defect Code')
plt.ylabel('Defect Code')
plt.xlabel('Defect Score')
plt.grid(True)
plt.savefig(r'data\split_train_test_data\with_delta_values\defect_score_optimization\defect_box_plots.png')
plt.show()
