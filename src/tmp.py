if __name__ == "__main__":
    import json
    accuracy_dict = {}
    for k in range(1, 18):
        with open(f"./trained_models/tl_pgd7_blocks{k}_info.json", "r") as f:
            accuracy_dict[k] = json.loads(f.read()).get("best_accuracy")

    with open("./trained_models/tl_pgd7_accuracy.json", "w") as f:
        f.write(json.dumps(accuracy_dict))