import json

def serialize(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(map(serialize, value)) + "]"
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{k} = {serialize(v)}" for k, v in value.items()) + " }"
            if value
            else "{}"
        )
    raise TypeError(f"Cannot encode to TOML: {type(value)}")

def emit_toml(cfg):
    d = cfg.to_dict()
    lines = [
        "# Prepared from a previous successful RF config; no automatic calibration.",
        "# This is a NEW physical channel session, with ORIGINAL PHY/radio gains.",
    ]
    for k, v in d.items():
        if k in ("phy", "radio", "methods", "conditions"):
            continue
        lines.append(f"{k} = {serialize(v)}")
    for key in ("phy", "radio"):
        lines += ["", f"[{key}]"]
        for k, v in d[key].items():
            lines.append(f"{k} = {serialize(v)}")
    for method in d["methods"]:
        lines += ["", "[[methods]]"]
        lines.append(f"name = {serialize(method['name'])}")
        if method["parameters"]:
            lines.append(f"parameters = {serialize(method['parameters'])}")
    for cond in d["conditions"]:
        lines += ["", "[[conditions]]"]
        for k, v in cond.items():
            if v is not None:
                lines.append(f"{k} = {serialize(v)}")
    return "\n".join(lines) + "\n"
