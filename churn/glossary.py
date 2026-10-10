"""Plain-English definitions used for the dashboard's tooltips and "Key terms" panels.

One place for every explanation, so a term reads the same wherever it appears.
Each entry is short (one or two sentences) because it is shown in a tooltip.
"""

GLOSSARY = {
    # --- Metrics ------------------------------------------------------------------
    "Precision": "Of the customers the model flags as churners, the share who really churn: TP / (TP + FP). "
                 "Low precision wastes retention budget on customers who would have stayed.",
    "Recall": "Of the customers who really churn, the share the model catches: TP / (TP + FN). "
              "Also called sensitivity or true positive rate. Low recall means lost customers.",
    "F1": "The harmonic mean of Precision and Recall: 2·P·R / (P + R). High only when both are high.",
    "Accuracy": "The share of all predictions that are correct. Misleading here: predicting 'stays' for everyone "
                "is already about 80% accurate.",
    "ROC_AUC": "Area under the ROC curve: the probability that a random churner gets a higher churn score than a "
               "random non-churner. 0.5 = coin flip, 1.0 = perfect ranking. Does not depend on the threshold.",
    "Average Precision": "Area under the Precision–Recall curve. Its no-skill baseline is the churn rate (~20%), "
                         "not 0.5.",
    "Log-loss": "The training loss of the classifiers: it penalises confident wrong probabilities heavily. "
                "Lower is better.",
    # --- Evaluation concepts -----------------------------------------------------------
    "Confusion matrix": "A 2×2 table of actual vs predicted classes: true negatives, false positives, false "
                        "negatives and true positives.",
    "True positive (TP)": "A churner the model correctly flags.",
    "False positive (FP)": "A loyal customer the model wrongly flags as a churner (a wasted retention offer).",
    "False negative (FN)": "A churner the model misses (a customer lost without any retention attempt).",
    "True negative (TN)": "A loyal customer the model correctly leaves alone.",
    "ROC curve": "True positive rate against false positive rate at every possible threshold.",
    "Precision–Recall curve": "Precision against Recall at every possible threshold; more informative than ROC "
                              "when the positive class is rare.",
    "Threshold": "The probability cut-off t: a customer is flagged as a churner when P(churn) ≥ t. Lowering t raises "
                 "Recall and usually lowers Precision.",
    "Operating point": "The point on a curve that corresponds to the threshold currently in use.",
    "Class imbalance": "One class is much rarer than the other (here ~20% churners), so accuracy is misleading and "
                       "models lean towards the majority class.",
    "Class weighting": "Making errors on the rare class count more during training, e.g. weight = n / (2 · n_class).",
    # --- Validation ---------------------------------------------------------------
    "Train/test split": "The data is split once: models learn from the training set and are evaluated on the "
                        "held-out test set they never saw.",
    "Stratification": "Splitting so that every part keeps the same churn rate as the whole dataset.",
    "Cross-validation (CV)": "The training set is split into k folds; each fold is held out once while a model is "
                             "trained on the others. Gives k scores instead of one.",
    "Mean ± std": "The average of the k fold scores, and their standard deviation: how much the score varies "
                  "with the sample of customers.",
    "Out-of-fold (OOF) prediction": "A prediction for a training customer from a CV model that did not see that "
                                    "customer during training.",
    "Data leakage": "Information from the evaluation data reaching the model during training, which makes scores "
                    "look better than they really are.",
    "Overfitting": "Fitting the training data so closely (including its noise) that performance on new data is "
                   "worse. Shows up as a training score much higher than the CV or test score.",
    "Generalisation gap": "The difference between the score on the training data and the score on unseen data.",
    "Paired comparison": "Comparing two models fold by fold on the same folds, so differences are not caused by "
                         "different samples.",
    "Selection bias (tuning)": "The best of many grid candidates looks slightly better on the folds it was chosen "
                               "on than it will on new data. Nested CV removes this.",
    "Nested cross-validation": "An outer CV loop for evaluation with a separate inner CV loop for tuning, so the "
                               "evaluation folds are never used to choose hyperparameters.",
    # --- Tuning ------------------------------------------------------------------
    "Hyperparameter": "A setting chosen before training (e.g. tree depth), as opposed to parameters the model "
                      "learns from data (e.g. split points).",
    "GridSearchCV": "Tries every combination in a grid of hyperparameter values, scores each with "
                    "cross-validation, and refits the best one on the full training set.",
    "Baseline": "The model with its documented default settings, before tuning.",
    # --- Models ---------------------------------------------------------------------
    "Bagging": "Training many models on bootstrap samples and averaging them (Random Forest). Mainly reduces "
               "variance.",
    "Gradient boosting": "Adding shallow trees one at a time, each fitted to the errors (gradients) of the "
                         "ensemble so far (XGBoost, LightGBM, CatBoost).",
    # --- Importance -------------------------------------------------------------------
    "Built-in importance": "Importance a tree model computes during training (impurity decrease, gain or "
                           "prediction change). Biased towards features with many split points.",
    "Permutation importance": "The drop in test ROC-AUC when one feature's values are shuffled. ≈ 0 means the "
                              "model does not use the feature.",
    "One-hot encoding": "Turning a category (e.g. Geography) into one 0/1 column per value.",
    "Standardisation": "Rescaling a numeric feature to mean 0 and standard deviation 1, using training-set "
                       "statistics only.",
}

# Hyperparameters shown in the dashboard, keyed by the name used in the model's settings.
HYPERPARAMETERS = {
    "max_iter": "Maximum solver iterations for Logistic Regression.",
    "C": "Inverse regularisation strength of Logistic Regression; smaller C = stronger penalty.",
    "n_estimators": "Number of trees. Random Forest averages them; boosting adds them one after another.",
    "iterations": "CatBoost's name for the number of boosting rounds (trees).",
    "max_depth": "Maximum depth of each tree. Deeper trees capture more interactions but overfit more easily "
                 "(None = grow until leaves are pure).",
    "depth": "CatBoost's tree depth (its trees are symmetric: one split per level).",
    "min_samples_leaf": "Minimum customers in a leaf of a Random Forest tree. Larger values smooth the trees "
                        "and reduce overfitting.",
    "num_leaves": "LightGBM's cap on leaves per tree (it grows trees leaf by leaf, not level by level).",
    "learning_rate": "Shrinks each new boosting tree's contribution. Smaller = more conservative; usually needs "
                     "more trees.",
    "subsample": "Share of training rows each boosting tree sees (row subsampling).",
    "subsample_freq": "How often LightGBM redraws the row subsample (1 = every round).",
    "colsample_bytree": "Share of features each boosting tree may use.",
    "reg_lambda": "L2 penalty on leaf weights; larger values shrink leaf predictions.",
    "l2_leaf_reg": "CatBoost's L2 penalty on leaf values.",
    "tree_method": "XGBoost's split-finding algorithm ('hist' bins features into histograms for speed).",
    "eval_metric": "Metric XGBoost reports during training; it does not change the fitted model.",
    "importance_type": "Which built-in importance the model reports ('gain' = loss reduction).",
    "random_state": "Seed for randomness, so results are reproducible.",
    "random_seed": "CatBoost's random seed.",
    "class_weight": "Weights for each class in the training loss ('balanced' up-weights churners).",
    "scale_pos_weight": "XGBoost's weight for the positive class (#stayed / #churned when balanced).",
    "auto_class_weights": "CatBoost's class weighting ('Balanced' up-weights churners).",
}

METRIC_HELP = {m: GLOSSARY[m] for m in ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC"]}

# --- Hebrew ---------------------------------------------------------------------------
# Display names: the Hebrew term, with the English technical term kept in parentheses where it is the
# term practitioners actually use.
TERM_NAMES_HE = {
    "Precision": "דיוק חיובי (Precision)", "Recall": "רגישות (Recall)", "F1": "מדד F1",
    "Accuracy": "דיוק כולל (Accuracy)", "ROC_AUC": "ROC-AUC", "Average Precision": "דיוק ממוצע (Average Precision)",
    "Log-loss": "פונקציית הפסד לוגריתמית (Log-loss)", "Confusion matrix": "מטריצת בלבול",
    "True positive (TP)": "חיובי אמיתי (TP)", "False positive (FP)": "חיובי שגוי (FP)",
    "False negative (FN)": "שלילי שגוי (FN)", "True negative (TN)": "שלילי אמיתי (TN)", "ROC curve": "עקומת ROC",
    "Precision–Recall curve": "עקומת Precision–Recall", "Threshold": "סף החלטה", "Operating point": "נקודת עבודה",
    "Class imbalance": "חוסר איזון בין מחלקות", "Class weighting": "שקלול מחלקות",
    "Train/test split": "חלוקה לאימון ובדיקה", "Stratification": "ריבוד (Stratification)",
    "Cross-validation (CV)": "אימות צולב (CV)", "Mean ± std": "ממוצע ± סטיית תקן",
    "Out-of-fold (OOF) prediction": "חיזוי מחוץ לקיפול (OOF)", "Data leakage": "דליפת מידע",
    "Overfitting": "התאמת יתר (Overfitting)", "Generalisation gap": "פער הכללה", "Paired comparison": "השוואה מזווגת",
    "Selection bias (tuning)": "הטיית בחירה (בכוונון)", "Nested cross-validation": "אימות צולב מקונן",
    "Hyperparameter": "היפר-פרמטר", "GridSearchCV": "GridSearchCV", "Baseline": "מודל בסיס (Baseline)",
    "Bagging": "Bagging", "Gradient boosting": "Gradient boosting", "Built-in importance": "חשיבות מובנית",
    "Permutation importance": "חשיבות פרמוטציה", "One-hot encoding": "קידוד One-hot", "Standardisation": "תקנון (Standardisation)",
}

GLOSSARY_HE = {
    "Precision": "מבין הלקוחות שהמודל סימן כנוטשים, כמה באמת נטשו: TP / (TP + FP). דיוק נמוך מבזבז תקציב שימור "
                 "על לקוחות שהיו נשארים בכל מקרה.",
    "Recall": "מבין הלקוחות שבאמת נטשו, כמה המודל תפס: TP / (TP + FN). נקרא גם שיעור חיובי אמיתי. רגישות נמוכה "
              "פירושה לקוחות שאבדו.",
    "F1": "הממוצע ההרמוני של Precision ו-Recall: ‎2·P·R / (P + R). גבוה רק כששניהם גבוהים.",
    "Accuracy": "שיעור התחזיות הנכונות מכלל התחזיות. מטעה כאן: חיזוי 'נשאר' לכולם כבר נותן כ-80% דיוק.",
    "ROC_AUC": "השטח מתחת לעקומת ROC: ההסתברות שלקוח נוטש אקראי יקבל ציון סיכון גבוה יותר מלקוח נאמן אקראי. "
               "0.5 = הטלת מטבע, 1.0 = דירוג מושלם. אינו תלוי בסף.",
    "Average Precision": "השטח מתחת לעקומת Precision–Recall. קו הבסיס שלו הוא שיעור הנטישה (כ-20%), ולא 0.5.",
    "Log-loss": "פונקציית ההפסד שהמסווגים ממזערים באימון: מענישה בחומרה הסתברויות בטוחות ושגויות. נמוך יותר = טוב יותר.",
    "Confusion matrix": "טבלה 2×2 של מחלקה אמיתית מול חזויה: שליליים אמיתיים, חיוביים שגויים, שליליים שגויים "
                        "וחיוביים אמיתיים.",
    "True positive (TP)": "לקוח נוטש שהמודל סימן נכון.",
    "False positive (FP)": "לקוח נאמן שהמודל סימן בטעות כנוטש (הצעת שימור מבוזבזת).",
    "False negative (FN)": "לקוח נוטש שהמודל פספס (לקוח שאבד בלי שום ניסיון שימור).",
    "True negative (TN)": "לקוח נאמן שהמודל השאיר בצדק בשקט.",
    "ROC curve": "שיעור החיוביים האמיתיים מול שיעור החיוביים השגויים, בכל סף אפשרי.",
    "Precision–Recall curve": "Precision מול Recall בכל סף אפשרי; אינפורמטיבית יותר מ-ROC כשהמחלקה החיובית נדירה.",
    "Threshold": "נקודת החיתוך t: לקוח מסומן כנוטש כאשר P(נטישה) ≥ t. הורדת t מעלה את ה-Recall ובדרך כלל "
                 "מורידה את ה-Precision.",
    "Operating point": "הנקודה על העקומה שמתאימה לסף שבשימוש כרגע.",
    "Class imbalance": "מחלקה אחת נדירה בהרבה מהשנייה (כאן כ-20% נוטשים), ולכן Accuracy מטעה והמודלים נוטים "
                       "למחלקת הרוב.",
    "Class weighting": "מתן משקל גבוה יותר לשגיאות על המחלקה הנדירה באימון, למשל משקל = n / (2 · n_class).",
    "Train/test split": "הנתונים מחולקים פעם אחת: המודלים לומדים מקבוצת האימון ונבחנים על קבוצת הבדיקה, שלא "
                        "ראו מעולם.",
    "Stratification": "חלוקה כך שבכל חלק נשמר אותו שיעור נטישה כמו בכלל הנתונים.",
    "Cross-validation (CV)": "קבוצת האימון מחולקת ל-k קיפולים; כל קיפול מוחזק בצד פעם אחת בזמן שהמודל מתאמן על "
                             "השאר. מתקבלים k ציונים במקום ציון אחד.",
    "Mean ± std": "ממוצע k ציוני הקיפולים וסטיית התקן שלהם: עד כמה הציון משתנה עם מדגם הלקוחות.",
    "Out-of-fold (OOF) prediction": "חיזוי עבור לקוח מקבוצת האימון, ממודל CV שלא ראה את הלקוח הזה באימון.",
    "Data leakage": "מידע מנתוני ההערכה שמגיע למודל בזמן האימון, וגורם לציונים להיראות טובים מהמציאות.",
    "Overfitting": "התאמה צמודה מדי לנתוני האימון (כולל הרעש שבהם), כך שהביצועים על נתונים חדשים גרועים יותר. "
                   "מתבטאת בציון אימון גבוה בהרבה מציון ה-CV או הבדיקה.",
    "Generalisation gap": "ההפרש בין הציון על נתוני האימון לבין הציון על נתונים שלא נראו.",
    "Paired comparison": "השוואת שני מודלים קיפול-קיפול על אותם קיפולים, כך שההבדלים לא נובעים ממדגמים שונים.",
    "Selection bias (tuning)": "המועמד הטוב מבין רבים בגריד נראה מעט טוב יותר על הקיפולים שעליהם נבחר מאשר יהיה "
                               "על נתונים חדשים. אימות צולב מקונן מבטל זאת.",
    "Nested cross-validation": "לולאת CV חיצונית להערכה ולולאת CV פנימית נפרדת לכוונון, כך שקיפולי ההערכה לעולם "
                               "לא משמשים לבחירת היפר-פרמטרים.",
    "Hyperparameter": "הגדרה שנבחרת לפני האימון (למשל עומק העץ), בניגוד לפרמטרים שהמודל לומד מהנתונים (למשל "
                      "נקודות פיצול).",
    "GridSearchCV": "מנסה כל צירוף בגריד של ערכי היפר-פרמטרים, מדרג כל אחד באימות צולב, ומאמן מחדש את הטוב ביותר "
                    "על כל קבוצת האימון.",
    "Baseline": "המודל עם הגדרות ברירת המחדל המתועדות שלו, לפני כוונון.",
    "Bagging": "אימון מודלים רבים על דגימות bootstrap ומיצוע שלהם (Random Forest). מקטין בעיקר שונות.",
    "Gradient boosting": "הוספת עצים רדודים אחד אחרי השני, שכל אחד מותאם לשגיאות (הגרדיאנטים) של האנסמבל עד כה "
                         "(XGBoost, LightGBM, CatBoost).",
    "Built-in importance": "חשיבות שמודל עצים מחשב בזמן האימון (ירידה באי-טוהר, gain או שינוי בחיזוי). מוטה "
                           "לטובת משתנים עם הרבה נקודות פיצול.",
    "Permutation importance": "הירידה ב-ROC-AUC של הבדיקה כאשר ערכי משתנה אחד מעורבבים. ≈ 0 פירושו שהמודל לא "
                              "משתמש במשתנה.",
    "One-hot encoding": "הפיכת קטגוריה (למשל Geography) לעמודת 0/1 אחת לכל ערך.",
    "Standardisation": "שינוי קנה מידה של משתנה מספרי לממוצע 0 וסטיית תקן 1, לפי סטטיסטיקות קבוצת האימון בלבד.",
}

HYPERPARAMETERS_HE = {
    "max_iter": "מספר איטרציות מרבי של הפותר ב-Logistic Regression.",
    "C": "ההופכי של עוצמת הרגולריזציה ב-Logistic Regression; C קטן יותר = ענישה חזקה יותר.",
    "n_estimators": "מספר העצים. Random Forest ממצע אותם; ב-boosting הם מתווספים אחד אחרי השני.",
    "iterations": "השם ב-CatBoost למספר סבבי ה-boosting (עצים).",
    "max_depth": "העומק המרבי של כל עץ. עצים עמוקים לוכדים יותר אינטראקציות אך נוטים יותר להתאמת יתר "
                 "(None = צמיחה עד עלים טהורים).",
    "depth": "עומק העץ ב-CatBoost (העצים שלו סימטריים: פיצול אחד לכל רמה).",
    "min_samples_leaf": "מספר הלקוחות המינימלי בעלה של עץ ב-Random Forest. ערכים גדולים מחליקים את העצים ומפחיתים "
                        "התאמת יתר.",
    "num_leaves": "התקרה של LightGBM למספר העלים בעץ (הוא מגדל עצים עלה אחר עלה, לא רמה אחר רמה).",
    "learning_rate": "מקטין את התרומה של כל עץ boosting חדש. קטן יותר = שמרני יותר; בדרך כלל דורש יותר עצים.",
    "subsample": "שיעור שורות האימון שכל עץ boosting רואה (דגימת שורות).",
    "subsample_freq": "באיזו תדירות LightGBM מגריל מחדש את דגימת השורות (1 = בכל סבב).",
    "colsample_bytree": "שיעור המשתנים שכל עץ boosting רשאי להשתמש בהם.",
    "reg_lambda": "ענישת L2 על משקלי העלים; ערכים גבוהים מכווצים את חיזויי העלים.",
    "l2_leaf_reg": "ענישת L2 של CatBoost על ערכי העלים.",
    "tree_method": "אלגוריתם מציאת הפיצולים של XGBoost ('hist' מחלק משתנים להיסטוגרמות לשם מהירות).",
    "eval_metric": "המדד ש-XGBoost מדווח במהלך האימון; אינו משנה את המודל המאומן.",
    "importance_type": "איזו חשיבות מובנית המודל מדווח ('gain' = הפחתת הפסד).",
    "random_state": "זרע אקראיות, כדי שהתוצאות יהיו ניתנות לשחזור.",
    "random_seed": "זרע האקראיות של CatBoost.",
    "class_weight": "משקלים לכל מחלקה בפונקציית ההפסד ('balanced' מגדיל את משקל הנוטשים).",
    "scale_pos_weight": "המשקל של XGBoost למחלקה החיובית (#נשארו / #נטשו במצב מאוזן).",
    "auto_class_weights": "שקלול המחלקות של CatBoost ('Balanced' מגדיל את משקל הנוטשים).",
}

METRIC_HELP_HE = {m: GLOSSARY_HE[m] for m in METRIC_HELP}


def define(term: str, lang: str = "en") -> str:
    """Definition for a glossary term or hyperparameter (KeyError if unknown, so typos fail tests)."""
    if lang == "he":
        return GLOSSARY_HE[term] if term in GLOSSARY_HE else HYPERPARAMETERS_HE[term]
    return GLOSSARY[term] if term in GLOSSARY else HYPERPARAMETERS[term]


def term_name(term: str, lang: str = "en") -> str:
    """How a glossary term is displayed (Hebrew names keep the English technical term where useful)."""
    return TERM_NAMES_HE[term] if lang == "he" else term.replace("_", "-")
