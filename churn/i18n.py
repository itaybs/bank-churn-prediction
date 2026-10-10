"""Interface text for the dashboard in English (primary) and Hebrew.

Covers navigation, page titles and subtitles, sidebar controls, shared labels and the
Executive Briefing page. Metric and statistical definitions live in churn.glossary;
the story game's text lives in churn/game_assets/i18n.js. Deeper analytical prose and
generated insights stay in English; Hebrew mode says so on those pages.

Hebrew text addresses the reader in the plural ("אתם"), which is gender-neutral.
"""

LANGUAGES = {"en": "English", "he": "עברית"}
RTL = {"he"}

PAGE_TITLES = {
    "en": [
        "Executive Briefing & Journey Map",
        "Project Overview & Data Inspection",
        "Preprocessing & Pipeline Architecture",
        "Model Explorer & Comparison",
        "Hyperparameter Tuning: Before vs After",
        "Cross-Validation & Robustness",
        "Feature Importance",
        "Interactive Churn Prediction Tool",
        "Methodology & Future Roadmap",
        "The Chief Retention Officer: A Machine Learning Decision Story",
    ],
    "he": [
        "תדריך מנהלים ומפת המסע",
        "סקירת הפרויקט ובדיקת הנתונים",
        "עיבוד מקדים וארכיטקטורת ה-Pipeline",
        "סייר המודלים והשוואה",
        "כוונון היפר-פרמטרים: לפני ואחרי",
        "אימות צולב ויציבות",
        "חשיבות משתנים",
        "כלי חיזוי נטישה אינטראקטיבי",
        "מתודולוגיה ומפת דרכים",
        "מנהל/ת השימור: סיפור החלטות בלמידת מכונה",
    ],
}

PAGE_SUBTITLES_HE = [
    "התחילו כאן: מה הפרויקט ביקש לעשות, מה הוא מצא, ואיך לנווט בין הראיות.",
    "הבעיה העסקית, הנתונים שמאחוריה, והבדיקות שמבטיחות שהנתונים בטוחים למידול.",
    "איך נתונים גולמיים הופכים לקלט למודל, צעד אחר צעד, ואיך התכנון מונע דליפת מידע.",
    "{n_models} מודלים, אותם {n_test} לקוחות בדיקה מוחזקים, והתצורה שנבחרה בסרגל הצד.",
    "GridSearchCV על קיפולי האימון: מה השתנה, האם זה עזר, ואיזה מודל הוא הטוב ביותר?",
    "אימות צולב מרובד ב-{folds} קיפולים על קבוצת האימון: כמה טוב, וכמה יציב, כל מודל?",
    "על אילו מאפייני לקוח המודלים נשענים, ועד כמה אפשר לסמוך על כל מדד חשיבות?",
    "הזינו פרופיל לקוח וצפו בסיכון הנטישה מתעדכן בזמן אמת.",
    "הסטטיסטיקה שמאחורי המדדים, הספים ושקלול המחלקות, ולאן הארכיטקטורה ממשיכה.",
    "נהלו את השימור בבנק מסחרי במשך חמישה שבועות. כל לחיצה היא החלטה; כל תוצאה אמיתית.",
]

STRINGS = {
    # --- Sidebar --------------------------------------------------------------------
    "language": {"en": "🌐 Language · שפה", "he": "🌐 שפה · Language"},
    "app_name": {"en": "📊 Churn Analytics", "he": "📊 אנליטיקת נטישה"},
    "app_caption": {"en": "Bank customer churn · Logistic Regression · Random Forest · XGBoost · LightGBM · CatBoost",
                    "he": "נטישת לקוחות בנק · Logistic Regression · Random Forest · XGBoost · LightGBM · CatBoost"},
    "navigate": {"en": "Navigate", "he": "ניווט"},
    "navigate_help": {"en": "Each section explains what was done, why, and how to read it.",
                      "he": "כל פרק מסביר מה נעשה, למה, ואיך לקרוא את התוצאות."},
    "model_config": {"en": "Model configuration", "he": "תצורת המודל"},
    "weighting": {"en": "Class balancing / weighting", "he": "איזון / שקלול מחלקות"},
    "weighting_help": {
        "en": "Balanced: errors on churners count more during training (inverse class frequency). Logistic "
              "Regression, Random Forest and LightGBM use class_weight='balanced'; XGBoost uses scale_pos_weight = "
              "#stayed / #churned from the training labels; CatBoost uses auto_class_weights='Balanced'. Models are "
              "retrained the first time you switch.",
        "he": "מאוזן: שגיאות על נוטשים נחשבות יותר באימון (ביחס הפוך לשכיחות המחלקה). Logistic Regression, "
              "Random Forest ו-LightGBM משתמשים ב-class_weight='balanced'; XGBoost ב-scale_pos_weight = "
              "#נשארו / #נטשו מתוויות האימון; CatBoost ב-auto_class_weights='Balanced'. המודלים מאומנים מחדש "
              "בפעם הראשונה שמחליפים.",
    },
    "None (equal weights)": {"en": "None (equal weights)", "he": "ללא (משקלים שווים)"},
    "Balanced (up-weight churners)": {"en": "Balanced (up-weight churners)", "he": "מאוזן (משקל גבוה לנוטשים)"},
    "strategy": {"en": "Threshold strategy", "he": "אסטרטגיית סף"},
    "strategy_help": {
        "en": "Default: P(churn) ≥ 0.5. CV-optimized: per-model threshold that maximised F1 in 5-fold "
              "cross-validation on the training set (no test data used). Custom: use the slider.",
        "he": "ברירת מחדל: P(נטישה) ≥ 0.5. מיטוב CV: סף לכל מודל שממקסם את F1 באימות צולב של 5 קיפולים על "
              "קבוצת האימון (בלי שימוש בנתוני הבדיקה). מותאם אישית: השתמשו במחוון.",
    },
    "Default (0.5 cutoff)": {"en": "Default (0.5 cutoff)", "he": "ברירת מחדל (סף 0.5)"},
    "CV-optimized (max F1)": {"en": "CV-optimized (max F1)", "he": "מיטוב CV (F1 מרבי)"},
    "Custom (slider)": {"en": "Custom (slider)", "he": "מותאם אישית (מחוון)"},
    "custom_threshold": {"en": "Decision threshold (custom)", "he": "סף החלטה (מותאם אישית)"},
    "custom_threshold_help": {
        "en": "Active only with the Custom strategy. A customer is flagged when P(churn) ≥ threshold. "
              "Lower threshold → more churners caught (Recall ↑) but more false alarms (Precision ↓).",
        "he": "פעיל רק באסטרטגיה המותאמת אישית. לקוח מסומן כאשר P(נטישה) ≥ הסף. סף נמוך יותר ← יותר נוטשים "
              "נתפסים (Recall ↑) אבל יותר אזעקות שווא (Precision ↓).",
    },
    "custom_warning": {
        "en": "Exploratory mode: choosing a threshold by watching test-set metrics is test-set tuning. "
              "Use the CV-optimized strategy for an honest estimate.",
        "he": "מצב חקירה: בחירת סף תוך צפייה במדדי קבוצת הבדיקה היא כוונון על קבוצת הבדיקה. "
              "לאומדן הוגן השתמשו באסטרטגיית מיטוב ה-CV.",
    },
    "dataset_caption": {
        "en": "Dataset: `{file}` · {n:,} customers · train {train:,} / test {test:,} · {folds}-fold CV · seed {seed}",
        "he": "נתונים: `{file}` · {n:,} לקוחות · אימון {train:,} / בדיקה {test:,} · CV ב-{folds} קיפולים · זרע {seed}",
    },
    "educational": {"en": "Educational project. Not validated for production banking decisions.",
                    "he": "פרויקט לימודי. לא אומת לקבלת החלטות בנקאיות אמיתיות."},
    "glossary": {"en": "📖 Glossary of terms", "he": "📖 מילון מונחים"},
    "dataset_error": {"en": "The dataset could not be loaded or validated: {error}",
                      "he": "לא ניתן היה לטעון או לאמת את הנתונים: {error}"},
    # --- Shared --------------------------------------------------------------------
    "section": {"en": "Section", "he": "פרק"},
    "what_we_did": {"en": "What we did", "he": "מה עשינו"},
    "why_it_matters": {"en": "Why it matters", "he": "למה זה חשוב"},
    "how_to_read": {"en": "How to read it", "he": "איך לקרוא"},
    "key_terms": {"en": "ℹ️ Key terms in this section", "he": "ℹ️ מונחי מפתח בפרק זה"},
    "english_note": {
        "en": "",
        "he": "ℹ️ הכותרות, ההגדרות וההסברים על המדדים בעברית. חלק מהניתוחים המפורטים והתובנות שמחושבות "
              "אוטומטית מוצגים באנגלית.",
    },
    # --- Executive briefing ---------------------------------------------------------
    "hero_eyebrow": {"en": "Executive briefing · bank customer churn", "he": "תדריך מנהלים · נטישת לקוחות בנק"},
    "hero_title": {"en": "{rate} of customers left the bank. Can we see it coming?",
                   "he": "{rate} מהלקוחות עזבו את הבנק. האם אפשר לראות את זה מגיע?"},
    "hero_body": {
        "en": "{churned:,} of {n:,} customers exited. This project builds and stress-tests {models} machine-learning "
              "models that rank every customer by churn risk, so a retention team can act <i>before</i> revenue "
              "walks out of the door.",
        "he": "{churned:,} מתוך {n:,} לקוחות עזבו. הפרויקט בונה ובוחן לעומק {models} מודלים של למידת מכונה "
              "שמדרגים כל לקוח לפי סיכון הנטישה שלו, כדי שצוות השימור יוכל לפעול <i>לפני</i> שההכנסות יוצאות "
              "מהדלת.",
    },
    "m1_tag": {"en": "The mission · where we are diving", "he": "המשימה · לאן אנחנו צוללים"},
    "m1_title": {"en": "Understand and predict churn", "he": "להבין ולחזות נטישה"},
    "m1_body": {
        "en": "Every customer who leaves takes their balances, products and future revenue with them. Retaining a "
              "customer is typically cheaper than acquiring a new one, so knowing <b>who is likely to leave</b> "
              "turns churn from a loss we report into a risk we can manage.",
        "he": "כל לקוח שעוזב לוקח איתו את היתרות, המוצרים וההכנסות העתידיות. שימור לקוח זול בדרך כלל מגיוס לקוח "
              "חדש, ולכן הידיעה <b>מי צפוי לעזוב</b> הופכת את הנטישה מהפסד שמדווחים עליו לסיכון שאפשר לנהל.",
    },
    "m2_tag": {"en": "The investigation · what we look for", "he": "החקירה · מה אנחנו מחפשים"},
    "m2_title": {"en": "Behavioural and demographic triggers", "he": "טריגרים התנהגותיים ודמוגרפיים"},
    "m2_body": {
        "en": "<p>Which customer signals separate leavers from stayers?</p><ul><li>Number of products held</li>"
              "<li>Activity status</li><li>Age and geography</li><li>Balance patterns</li></ul>",
        "he": "<p>אילו אותות מבדילים בין לקוחות שעוזבים ללקוחות שנשארים?</p><ul><li>מספר המוצרים</li>"
              "<li>סטטוס פעילות</li><li>גיל ומדינה</li><li>דפוסי יתרה</li></ul>",
    },
    "m3_tag": {"en": "Success criteria · how we will know", "he": "מדדי הצלחה · איך נדע"},
    "m3_title": {"en": "Rigour, not flattering accuracy", "he": "קפדנות, לא דיוק מחמיא"},
    "m3_body": {
        "en": "<p>~80% accuracy is free (predict \"stays\" for everyone), so it proves nothing. Instead:</p>"
              "<ul><li><b>ROC-AUC</b>: ranking quality, threshold-free</li><li><b>5-fold cross-validation</b>: "
              "mean ± std, not one lucky split</li><li><b>Threshold tuning</b> on training data to trade Precision "
              "for Recall</li></ul>",
        "he": "<p>דיוק של כ-80% מתקבל בחינם (לחזות \"נשאר\" לכולם), ולכן הוא לא מוכיח דבר. במקום זאת:</p>"
              "<ul><li><b>ROC-AUC</b>: איכות הדירוג, ללא תלות בסף</li><li><b>אימות צולב ב-5 קיפולים</b>: "
              "ממוצע ± סטיית תקן, לא חלוקה אחת ברת-מזל</li><li><b>כוונון סף</b> על נתוני האימון, כדי לאזן בין "
              "Precision ל-Recall</li></ul>",
    },
    "headline": {"en": "Headline results", "he": "תוצאות מרכזיות"},
    "k_selected": {"en": "Selected model", "he": "המודל שנבחר"},
    "k_selected_help": {"en": "Highest mean ROC-AUC in 5-fold cross-validation on the training set (the test set only confirms).",
                        "he": "ה-ROC-AUC הממוצע הגבוה ביותר באימות צולב של 5 קיפולים על קבוצת האימון (קבוצת הבדיקה רק מאשרת)."},
    "k_auc": {"en": "CV ROC-AUC (± {std})", "he": "ROC-AUC ב-CV (± {std})"},
    "k_auc_help": {"en": "Mean over {folds} folds; ± {std} is the standard deviation across folds. ",
                   "he": "ממוצע על פני {folds} קיפולים; ± {std} היא סטיית התקן בין הקיפולים. "},
    "k_recall": {"en": "Churners caught (Recall)", "he": "נוטשים שנתפסו (Recall)"},
    "k_recall_help": {"en": "Test-set Recall at the CV-optimized threshold ({t}) vs the default 0.5. ",
                      "he": "Recall בקבוצת הבדיקה בסף שמוטב ב-CV ({t}) לעומת ברירת המחדל 0.5. "},
    "k_precision": {"en": "Flags that are real churners", "he": "סימונים שהם נוטשים אמיתיים"},
    "k_precision_help": {"en": "Test-set Precision at the CV-optimized threshold: the price of catching more churners. ",
                         "he": "Precision בקבוצת הבדיקה בסף שמוטב ב-CV: המחיר של תפיסת יותר נוטשים. "},
    "pp_vs_default": {"en": "{d:+.0f} pp vs 0.5 cutoff", "he": "{d:+.0f} נק' אחוז לעומת סף 0.5"},
    "live_caption": {
        "en": "Live values for the current sidebar configuration (weighting: {w}). Recall and Precision are measured "
              "on the untouched test set.",
        "he": "ערכים חיים לתצורה הנוכחית בסרגל הצד (שקלול: {w}). Recall ו-Precision נמדדים על קבוצת הבדיקה "
              "שלא נגעו בה.",
    },
    "triggers": {"en": "What the data says: churn triggers", "he": "מה הנתונים אומרים: טריגרים לנטישה"},
    "triggers_intro": {
        "en": "Churn rate by segment in the **training set** (segments with fewer than {n} customers excluded). "
              "These are associations, not proven causes.",
        "he": "שיעור נטישה לפי פלח ב**קבוצת האימון** (פלחים עם פחות מ-{n} לקוחות לא נכללו). אלה מתאמים, "
              "לא סיבות מוכחות.",
    },
    "t_products": {"en": "Products held", "he": "מוצרים"},
    "t_activity": {"en": "Activity status", "he": "סטטוס פעילות"},
    "t_age": {"en": "Age band", "he": "קבוצת גיל"},
    "t_geo": {"en": "Geography", "he": "מדינה"},
    "products_1": {"en": "1 product", "he": "מוצר אחד"},
    "products_n": {"en": "{v} products", "he": "{v} מוצרים"},
    "inactive": {"en": "Inactive", "he": "לא פעילים"},
    "active": {"en": "Active", "he": "פעילים"},
    "age_v": {"en": "Age {v}", "he": "גיל {v}"},
    "geo_France": {"en": "France", "he": "צרפת"},
    "geo_Germany": {"en": "Germany", "he": "גרמניה"},
    "geo_Spain": {"en": "Spain", "he": "ספרד"},
    "trigger_line": {
        "en": "churn for <b>{hi}</b> (n = {n:,})<br>vs {lo_rate} for {lo} · <b>{ratio:.1f}×</b>",
        "he": "נטישה אצל <b>{hi}</b> (n = {n:,})<br>לעומת {lo_rate} אצל {lo} · <b>{ratio:.1f}×</b>",
    },
    "journey": {"en": "Analytical journey map", "he": "מפת המסע האנליטי"},
    "journey_intro": {
        "en": "Six stages from raw data to a decision tool. Select a stage to see what was done and what was "
              "measured, then jump to the full evidence.",
        "he": "שישה שלבים מנתונים גולמיים ועד כלי החלטה. בחרו שלב כדי לראות מה נעשה ומה נמדד, ואז קפצו "
              "לראיות המלאות.",
    },
    "journey_help": {"en": "Each stage links to the dashboard section with the full analysis.",
                     "he": "כל שלב מקשר לפרק בלוח הבקרה עם הניתוח המלא."},
    "stages": {
        "en": ["Data Inspection", "Preprocessing", "Base Models", "Advanced Boosters", "Tuning", "Final Deployment"],
        "he": ["בדיקת נתונים", "עיבוד מקדים", "מודלי בסיס", "מודלי Boosting", "כוונון", "הטמעה סופית"],
    },
    "step": {"en": "Step", "he": "שלב"},
    "what_we_did_stage": {"en": "what we did", "he": "מה עשינו"},
    "what_we_measured": {"en": "What we measured", "he": "מה מדדנו"},
    "open_section": {"en": "Open Section {n} →", "he": "← מעבר לפרק {n}"},
    "go_to": {"en": "Go to: {t}", "he": "מעבר אל: {t}"},
    "stage_details": {
        "en": [
            ("Validated 10,000 customers × 13 columns: no missing values, no duplicates, binary target. Found the one "
             "material issue: class imbalance.", "Churn rate {base}: accuracy alone is misleading."),
            ("Dropped identifiers, one-hot encoded Geography and Gender, scaled numerics for the linear model, and put "
             "everything inside a scikit-learn Pipeline after a stratified 80/20 split.",
             "Train churn {train} vs test {test}; preprocessing learns from training rows only (no leakage)."),
            ("Established reference points: an interpretable linear model and a bagging ensemble of deep trees.",
             "CV ROC-AUC: Logistic Regression {lr}, Random Forest {rf}. Non-linear patterns matter."),
            ("Added three gradient-boosting implementations that build shallow trees sequentially, each correcting "
             "the errors of the last.", "CV ROC-AUC: {boosters}. The boosters are practically tied at the top."),
            ("Searched hyperparameter grids with GridSearchCV on the training folds only, and compared before vs after "
             "on cross-validation and on the untouched test set.", "{tuning}"),
            ("Selected the model on cross-validation, chose a decision threshold from out-of-fold predictions, and "
             "exposed everything in an interactive scoring tool. Tuned pipelines are saved for reuse.",
             "{best} at threshold {t} catches {recall} of test churners. Ready for exploration and teaching; not "
             "validated for production banking decisions."),
        ],
        "he": [
            ("אימתנו 10,000 לקוחות × 13 עמודות: אין ערכים חסרים, אין כפילויות, משתנה מטרה בינארי. נמצאה בעיה "
             "מהותית אחת: חוסר איזון בין מחלקות.", "שיעור נטישה {base}: מדד ה-Accuracy לבדו מטעה."),
            ("הסרנו מזהים, קודדנו את Geography ו-Gender ב-one-hot, תקננו משתנים מספריים עבור המודל הלינארי, "
             "והכנסנו הכול ל-Pipeline של scikit-learn אחרי חלוקה מרובדת 80/20.",
             "נטישה באימון {train} לעומת בבדיקה {test}; העיבוד המקדים לומד משורות האימון בלבד (אין דליפת מידע)."),
            ("קבענו נקודות ייחוס: מודל לינארי פרשני ואנסמבל bagging של עצים עמוקים.",
             "ROC-AUC ב-CV: Logistic Regression {lr}, Random Forest {rf}. דפוסים לא-לינאריים חשובים."),
            ("הוספנו שלושה מימושי gradient boosting שבונים עצים רדודים ברצף, וכל אחד מתקן את שגיאות קודמו.",
             "ROC-AUC ב-CV: {boosters}. מודלי ה-boosting כמעט שקולים בצמרת."),
            ("חיפשנו בגרידים של היפר-פרמטרים עם GridSearchCV על קיפולי האימון בלבד, והשווינו לפני ואחרי באימות "
             "צולב ובקבוצת הבדיקה שלא נגעו בה.", "{tuning}"),
            ("בחרנו את המודל לפי אימות צולב, קבענו סף החלטה מחיזויי OOF, והנגשנו הכול בכלי ניקוד אינטראקטיבי. "
             "ה-Pipelines המכווננים נשמרים לשימוש חוזר.",
             "{best} בסף {t} תופס {recall} מהנוטשים בקבוצת הבדיקה. מוכן לחקירה ולהוראה; לא אומת להחלטות בנקאיות "
             "אמיתיות."),
        ],
    },
    "tuning_result": {
        "en": "Random Forest CV ROC-AUC {before} → {after} after GridSearchCV; XGBoost's defaults were already near-optimal.",
        "he": "ה-ROC-AUC ב-CV של Random Forest עלה מ-{before} ל-{after} אחרי GridSearchCV; ברירות המחדל של XGBoost "
              "כבר היו קרובות לאופטימום.",
    },
    "tuning_missing": {"en": "Run `python main.py` to produce the tuning results.",
                       "he": "הריצו `python main.py` כדי להפיק את תוצאות הכוונון."},
    "how_to_read_dashboard": {
        "en": "**How to read this dashboard.** Use the sidebar to move between sections and to change the class "
              "weighting and decision threshold; every chart updates live. Hover any metric or column header for a "
              "definition, and open **Key terms** panels or the sidebar **Glossary** for more.",
        "he": "**איך לקרוא את לוח הבקרה.** השתמשו בסרגל הצד כדי לעבור בין פרקים ולשנות את שקלול המחלקות ואת סף "
              "ההחלטה; כל גרף מתעדכן בזמן אמת. רחפו מעל כל מדד או כותרת עמודה לקבלת הגדרה, ופתחו את חלוניות "
              "**מונחי המפתח** או את **מילון המונחים** בסרגל הצד להרחבה.",
    },
    # --- Story game page --------------------------------------------------------------
    "game_caption": {
        "en": "Everything below is interactive and runs instantly in your browser. Customers are real held-out test "
              "customers (fictional names), scored live by every model, and their outcomes are real. Customer values, "
              "incentive costs and success rates are game assumptions. Current sidebar weighting: **{w}**.",
        "he": "כל מה שמתחת אינטראקטיבי ורץ מיידית בדפדפן. הלקוחות הם לקוחות בדיקה אמיתיים שהוחזקו בצד (בשמות "
              "בדויים), כל המודלים מדרגים אותם בזמן אמת, והתוצאות שלהם אמיתיות. ערך הלקוחות, עלויות התמריצים "
              "ושיעורי ההצלחה הם הנחות של המשחק. השקלול הנוכחי בסרגל הצד: **{w}**.",
    },
    "game_how": {"en": "How the game works, and what is real", "he": "איך המשחק עובד, ומה בו אמיתי"},
}


def tr(key: str, lang: str = "en", **kwargs):
    """Text for key in lang (falls back to English), formatted with kwargs. Unknown keys raise KeyError."""
    entry = STRINGS[key]
    value = entry.get(lang, entry["en"])
    return value.format(**kwargs) if kwargs and isinstance(value, str) else value


def page_labels(lang: str = "en") -> list[str]:
    """Navigation labels, e.g. '3. Model Explorer & Comparison'."""
    return [f"{i}. {title}" for i, title in enumerate(PAGE_TITLES[lang])]
