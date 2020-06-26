from pyecharts.charts import Bar, Grid, Page
from pyecharts.render import make_snapshot
from pyecharts import options as opts
from snapshot_pyppeteer import snapshot

from typing import Sequence, Dict, Any, List
from enum import Enum, unique
import os

from utils import load_json


@unique
class AttackName(Enum):
    accuracy = -1
    FGSM = 0
    PGD = 1
    DeepFool = 2
    cw2 = 3


@unique
class AttackNameIncludeAccuracy(Enum):
    accuracy = 0
    FGSM = 1
    PGD = 2
    DeepFool = 3
    cw2 = 4


class DrawTool:

    def __init__(self):
        self.page = Page(layout=Page.DraggablePageLayout)

    def add_bar(self, xaxis: Sequence, yaxis_seq_name: Sequence[str],
                yaxis_seq: Sequence[Sequence], title: str = None, **kwargs) -> None:

        bar = Bar().set_global_opts(
            title_opts=opts.TitleOpts(title=title)
        )
        bar.add_xaxis(xaxis)

        for name, yaxis in zip(yaxis_seq_name, yaxis_seq):
            yaxis = [float(format(x, ".2f")) for x in yaxis]
            bar.add_yaxis(
                name,
                yaxis,
            )

        self.page.add(bar)

    def render(self, save_path=None, **kwargs) -> None:
        self.page.render(save_path, **kwargs)


def cal_adv_success(data: Dict[str, Any]) -> Dict[str, Any]:
    """calculate success of attack"""
    accuracy_data = data["accuracy"]
    for attack_name, attack_data in data.items():
        if attack_name != "accuracy":
            for model_name, model_accuracy in accuracy_data.items():
                adv_accuracy = attack_data[model_name]
                attack_data[model_name] = 1 - adv_accuracy / model_accuracy

    return data


def get_current_dir_json(dir_path: str) -> Dict[str, Any]:
    """load json file from current directory"""
    data = dict()

    file_list = os.listdir(dir_path)
    for file in file_list:
        prefix, suffix = os.path.splitext(file)
        if suffix == ".json":
            data[prefix] = load_json(os.path.join(dir_path, file))

    # return cal_adv_success(data)
    return data


def short_attack_name(attack_name_list: Sequence) -> Sequence:
    new_attack_list = []
    for attack_name in attack_name_list:
        if attack_name == "DeepFool":
            new_attack_list.append("DeepFool")
        elif attack_name == "FastGradientMethod":
            new_attack_list.append("FGSM")
        elif attack_name == "ProjectedGradientDescent":
            new_attack_list.append("PGD")
        elif attack_name == "CarliniL2Method":
            new_attack_list.append("cw2")
        elif attack_name == "accuracy":
            new_attack_list.append("benign")

    return new_attack_list


def draw_all(root_dir: str, title: str, save_path: str, draw_success=False) -> None:

    def insert_data(xaxis: List, yaxis: Dict, name: str, info: dict, index: int):
        xaxis[index] = name
        for model_name, adv_success in info.items():
            if yaxis.get(model_name):
                yaxis[model_name][index] = adv_success
            else:
                yaxis[model_name] = [0 for i in range(5)]
                yaxis[model_name][index] = adv_success

    epsilon = ["normal", "005", "010", "015"]
    draw_tool = DrawTool()

    for eps in epsilon:
        data = get_current_dir_json(os.path.join(root_dir, eps))

        if draw_success:
            data = cal_adv_success(data)
            data.pop("accuracy")
            enum_ins = AttackName
        else:
            enum_ins = AttackNameIncludeAccuracy

        xaxis = ["" for i in range(len(data))]
        yaxis = dict()
        for name, info in data.items():
            # put accuracy to first row
            if name == "accuracy":
                insert_data(xaxis, yaxis, name, info, enum_ins.accuracy.value)
            elif name == "FastGradientMethod":
                insert_data(xaxis, yaxis, name, info, enum_ins.FGSM.value)
            elif name == "ProjectedGradientDescent":
                insert_data(xaxis, yaxis, name, info, enum_ins.PGD.value)
            elif name == "DeepFool":
                insert_data(xaxis, yaxis, name, info, enum_ins.DeepFool.value)
            elif name == "CarliniL2Method":
                insert_data(xaxis, yaxis, name, info, enum_ins.cw2.value)
        yaxis_seq = []
        yaxis_seq_name = []
        for name, adv_success_list in yaxis.items():
            yaxis_seq.append(adv_success_list)
            yaxis_seq_name.append(name)

        xaxis = short_attack_name(xaxis)

        if eps == "normal":
            _title = f"{title}(normal training)"
        else:
            _title = f"{title}(pgd{eps} training)"
        draw_tool.add_bar(
            xaxis=xaxis,
            yaxis_seq_name=yaxis_seq_name,
            yaxis_seq=yaxis_seq,
            title=_title
        )

    draw_tool.render(f"{save_path}")

if __name__ == '__main__':

    result_dir = "F:/ai lab/robust tl/result/"

    draw_all(os.path.join(result_dir, "student"), "accuracy of model", "student_accuracy.html")
    draw_all(os.path.join(result_dir, "student"), "success of adv", "student_success.html", draw_success=True)

    draw_all(os.path.join(result_dir, "teacher"), "accuracy of model", "teacher_accuracy.html")
    draw_all(os.path.join(result_dir, "teacher"), "success of adv", "teacher_success.html", draw_success=True)



