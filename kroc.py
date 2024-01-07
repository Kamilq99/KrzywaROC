#Nim przetestuje się działaniue kodu należy wpisać pip install matplotlib scikit-learn, aby zainstalować bibliotekę matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Dostarczone dane
TP = [193, 188, 185, 181, 173, 162, 134, 101, 72, 42, 32, 24, 23, 22, 22, 22, 21, 21, 21, 0]
FP = [7, 12, 15, 19, 27, 38, 66, 99, 128, 158, 168, 176, 177, 178, 178, 178, 179, 179, 179, 200]

# Oblicz czułość i swoistość
czulosc = [tp / (tp + fn) for tp, fn in zip(TP, [200] * len(TP))]
swoistosc = [tn / (tn + fp) for tn, fp in zip([0] * len(FP), FP)]

# Oblicz krzywą ROC
fpr, tpr, _ = roc_curve([1]*len(TP) + [0]*len(FP), czulosc + swoistosc)
roc_auc = auc(fpr, tpr)

# Narysuj krzywą ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Swoistość')
plt.ylabel('Czułość')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()