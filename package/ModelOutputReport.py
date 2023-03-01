'''
# -*- coding: utf-8 -*-
# @Author: Ang Jian Hwee <angjianhwee@gmail.com>
# @Date:   2023-03-01 18:07:26
# @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
# @Last Modified time: 2023-03-01 18:31:19
'''
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, average_precision_score
import pprint

class ModelOutputReport:
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
        self.report = {}
        self.report['accuracy'] = self._get_accuracy()
        self.report['confusion_matrix'] = self._get_confusion_matrix()
        self.report['classification_report'] = self._get_classification_report()
        self.report['roc_auc_score'] = self._get_roc_auc_score()
        self.report['f1_score'] = self._get_f1_score()
        self.report['precision_score'] = self._get_precision_score()
        self.report['recall_score'] = self._get_recall_score()
        self.report['roc_curve'] = self._get_roc_curve()
        self.report['precision_recall_curve'] = self._get_precision_recall_curve()
        self.report['average_precision_score'] = self._get_average_precision_score()

    def __repr__(self):
        return str(self.report)


    def _get_accuracy(self):
        return accuracy_score(self.y_test, self.y_pred)

    def _get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def _get_classification_report(self):
        return classification_report(self.y_test, self.y_pred)

    def _get_roc_auc_score(self):
        return roc_auc_score(self.y_test, self.y_pred)

    def _get_f1_score(self):
        return f1_score(self.y_test, self.y_pred)

    def _get_precision_score(self):
        return precision_score(self.y_test, self.y_pred)

    def _get_recall_score(self):
        return recall_score(self.y_test, self.y_pred)

    def _get_roc_curve(self):
        return roc_curve(self.y_test, self.y_pred)

    def _get_precision_recall_curve(self):
        return precision_recall_curve(self.y_test, self.y_pred)

    def _get_average_precision_score(self):
        return average_precision_score(self.y_test, self.y_pred)

    def print_report(self):
        for key, value in self.report.items():
            if key == 'classification_report':
                continue
            if hasattr(value, '__iter__'):
                print(f"{key}:")
                try:
                    pprint.pprint(value.round(4))
                except:
                    pprint.pprint(value)
            else:
                print(f"{key:<20s}: {value: >8.4f}")
            print()
        print(f"classification_report:\n{self.report['classification_report']}")
        

if __name__ == '__main__':
    sample_y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # sample_y_test different
    sample_y_test = [0, 1, 1, 1, 1, 1, 0, 1, 0, 1]

    model_output_report = ModelOutputReport(sample_y_pred, sample_y_test)
    # print(vars(model_output_report))
    model_output_report.print_report()