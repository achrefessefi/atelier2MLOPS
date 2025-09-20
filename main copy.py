# main.py
import argparse
import json
import pandas as pd

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

def cmd_prepare(args):
    bundle = prepare_data(
        csv_path=args.csv,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        drop_cols=args.drop_cols,
        build_churn_from_playtime_2w=not args.no_build_churn,
    )
    print(f"Train rows: {bundle.X_train.shape[0]} | Test rows: {bundle.X_test.shape[0]}")
    print(f"Features — numeric: {len(bundle.numeric)}, categorical: {len(bundle.categorical)}, boolean: {len(bundle.boolean)}")
    # sanity: show the preprocessing schema columns
    print("Numeric:", bundle.numeric[:15], "..." if len(bundle.numeric) > 15 else "")
    print("Categorical:", bundle.categorical[:15], "..." if len(bundle.categorical) > 15 else "")
    print("Boolean:", bundle.boolean[:15], "..." if len(bundle.boolean) > 15 else "")

def cmd_train(args):
    bundle = prepare_data(
        csv_path=args.csv,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        drop_cols=args.drop_cols,
        build_churn_from_playtime_2w=not args.no_build_churn,
    )
    model = train_model(bundle)
    save_model(model, args.out)
    print(f"Model saved to {args.out}")

    metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n", metrics["report"])

def cmd_evaluate(args):
    model = load_model(args.model)
    bundle = prepare_data(
        csv_path=args.csv,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        drop_cols=args.drop_cols,
        build_churn_from_playtime_2w=not args.no_build_churn,
    )
    metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n", metrics["report"])

def cmd_predict(args):
    model = load_model(args.model)
    df = pd.read_csv(args.csv)

    # Mirror the leakage-protection from prepare_data
    for col in [args.target, "playtime_2weeks", "median_playtime_2weeks", "game_id", "name", "release_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    preds = model.predict(df)
    out = args.out or "predictions.csv"
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    print(f"Predictions saved to {out}")

def build_parser():
    p = argparse.ArgumentParser(description="Churn classification pipeline (preprocessing done in preparation).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_io(sp):
        sp.add_argument("--csv", required=True, help="Path to CSV data file")
        sp.add_argument("--target", default="churn", help="Target column name (default: churn)")
        sp.add_argument("--test-size", type=float, default=0.2)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--drop-cols", nargs="*", default=[], help="Extra columns to drop from features")
        sp.add_argument("--no-build-churn", action="store_true", help="Do NOT auto-build 'churn' from playtime_2weeks")

    sp = sub.add_parser("prepare", help="Prepare: split, infer columns, and BUILD PREPROCESSOR")
    common_io(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="Train and save model (uses preprocessor from preparation)")
    common_io(sp)
    sp.add_argument("--out", default="churn_model.joblib", help="Output model path")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Load model and evaluate on freshly prepared split")
    common_io(sp)
    sp.add_argument("--model", required=True, help="Path to saved model")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("predict", help="Predict labels for a new CSV (no target needed)")
    sp.add_argument("--csv", required=True, help="Path to CSV without target")
    sp.add_argument("--model", required=True, help="Path to saved model")
    sp.add_argument("--out", default="predictions.csv", help="Where to save predictions CSV")
    sp.add_argument("--target", default="churn")
    sp.set_defaults(func=cmd_predict)

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)