/* The Chief Retention Officer: interface text in English (primary) and Hebrew.
 *
 * Values are strings or small functions for text with numbers. Both languages define exactly the
 * same keys (tests/test_story_game.py checks this under Node). Hebrew addresses the player in the
 * plural, which is gender-neutral; customer outcomes are phrased with nouns for the same reason.
 * Persona names, titles, stories and dialogue come from the Python payload (churn/story_game.py).
 */
(function (root) {
  "use strict";

  const en = {
    dir: "ltr",
    eyebrow: "A machine learning decision story",
    title: "🏦 The Chief Retention Officer",
    back: "← Back", skip: "Skip story", next: "Next →", startWeek1: "Start week 1 →",
    weekOf: (w, n) => `Week ${w} of ${n}`,
    deskTitle: "Five customers are on your desk",
    storyBtn: "📖 Story",
    hudBudget: "Budget left", hudCommitted: "Committed",
    targeted: (n) => `${n} customer${n === 1 ? "" : "s"} targeted`,
    hudExpSave: "Model-expected savings", hudExpSaveS: (m) => `if ${m}'s risks are right`,
    hudExpRoi: "Expected ROI", hudRoiS: "(savings − spend) / spend",
    hudSeason: "Season total", hudSeasonS: "net value over revealed weeks",
    hudRealSave: "✅ Real savings", hudRealSaveS: (w, l) => `wasted ${w} · lost ${l}`,
    hudRealRoi: "📈 Real ROI", netShort: (n) => `net ${n}`, na: "n/a",
    crystal: "🧠 Your crystal ball: choose a model",
    thresholdTitle: "🎚️ Decision threshold",
    useOpt: "Use model's optimum", default50: "Default 50%", autoIncLabel: "Incentive for flagged customers",
    targetAll: "🎯 Target all flagged", clear: "Clear",
    thresholdHint: "Customers with risk ≥ threshold are <b>flagged</b> (glowing red). Lower it to catch more churners (Recall ↑), raise it to waste less on loyal ones (Precision ↑).",
    scaleTitle: (n) => `🔭 This rule at scale: all ${n} test customers`,
    readyText: "Ready? Outcomes are the <i>real</i> results of these held-out customers.",
    revealBtn: "🎲 Reveal what happened", openDebrief: "📋 Open the debrief",
    revealHint: "Click a card to talk to the customer, then pick an incentive with the buttons.",
    revealedHint: "Outcomes revealed. Open the debrief for the lessons of the week.",
    riskHigh: "😠 HIGH RISK", riskWavering: "😐 WAVERING", riskLoyal: "😊 LOYAL",
    chipMeta: (auc, opt) => `ROC-AUC ${auc} · optimum ${opt}`,
    modelHint: (m, p) => `${m} ranks a random churner above a random loyal customer <b>${p}</b> of the time (cross-validated). ⭐ = selected by cross-validation.`,
    optimumOf: (m, p) => `${m} optimum: ${p}`,
    churnRisk: "Churn risk",
    geo: { France: "France", Germany: "Germany", Spain: "Spain" },
    facts: (c, geo, balance) => `Age ${c.age} · ${geo} · ${c.products} product${c.products > 1 ? "s" : ""} · ${c.active ? "active" : "inactive"} · ${c.tenure} yrs<br>Balance ${balance}`,
    valueAtStake: "Value at stake", noAction: "No action", none: "none",
    actTitle: (name, cost, succ) => `${name}: ${cost}, ${succ} success`,
    overBudget: "Over budget! Free up money elsewhere first.",
    scaleIncName: (icon, name) => `${icon} ${name} for every flagged customer`,
    sFlagged: "Flagged", sFlaggedS: (tp, fp) => `${tp} churners + ${fp} loyal`,
    sRecall: "Recall", sRecallS: (tp, all) => `${tp} of ${all} churners caught`,
    sPrecision: "Precision", sPrecisionS: (tp, f) => `${tp} of ${f} flags correct`,
    sWasted: "Wasted (FP)", sWastedS: (fp) => `${fp} loyal customers paid`,
    sMissed: "Missed (FN)", sMissedS: "churners never contacted",
    sNet: "Net value", sNetS: (sav, sp) => `savings ${sav} − spend ${sp}`,
    scaleHint: (m, t) => `Real test-set counts for ${m} at threshold ${t}. Move the slider or switch model and watch Precision and Recall trade places.`,
    resTP: (x) => `🚪 Was leaving · ✅ reached in time: expected saving ${x}`,
    resFN: (x) => `🚪 Left the bank · ❌ missed: lost ${x}`,
    resFP: (x) => `🏠 Was staying · ⚠️ ${x} wasted`,
    resTN: "🏠 Stayed · 👍 correctly left alone",
    list: (names) => names.join(", "),
    lThreshold: (t, names, scores, maxT, many) => `🎚️ <b>Your threshold (${t}) was too strict for ${names}</b>: scored ${scores}, but left. A threshold at or below ${maxT} would have flagged ${many ? "all of them" : "this customer"}. That is Recall: churners slip through when the bar is too high.`,
    lFalseFlags: (names, many) => `🎯 <b>${names} ${many ? "were" : "was"} flagged but stayed.</b> That is the Precision cost of your threshold: alarms on loyal customers.`,
    lWasted: (x, names) => `⚠️ <b>${x} wasted</b> on ${names}: false positives, incentives for customers who were staying anyway.`,
    lLost: (x, names) => `❌ <b>${x} lost</b> because ${names} left without any retention attempt: false negatives.`,
    lModelWeek: (top, diff, cur) => `🧠 <b>Model choice this week:</b> following ${top}'s risks would have earned ${diff} more than following ${cur}'s.`,
    lModelPaid: (cur, net) => `🧠 <b>Your model choice paid off this week:</b> no other model's plan beat ${cur}'s (${net}).`,
    lScale: (inc, n, bestNet, best, worstNet, worst) => `📊 <b>At scale, models matter:</b> a ${inc.toLowerCase()} campaign across ${n} customers earns ${bestNet} with ${best} but ${worstNet} with ${worst}, each at its optimal threshold. One week of five customers is noisy; thousands of decisions are not.`,
    debEyebrow: (w, m, t) => `Week ${w} debrief · model ${m} · threshold ${t}`,
    victory: "Victory", defeat: "Defeat",
    scoreLine: (s) => `Success score ${s}/100 = your net value as a share of the best achievable this week.`,
    kSavings: "✅ True savings", kSavingsS: "churners reached × success × value",
    kWasted: "⚠️ Wasted budget", kWastedS: "false positives",
    kLost: "❌ Lost revenue", kLostS: "missed churners + failed incentives",
    kRoi: "📈 Net ROI",
    weekTeaches: "What the week teaches", youVs: "You vs the benchmarks (this week)",
    you: "You", planOf: (m) => `${m}'s plan`, hindsight: "Perfect hindsight",
    everyModel: "Every model's own plan, real outcomes",
    foot: "A model's plan maximises expected value from its own risk scores within the budget. Hindsight knows who left.",
    backBoard: "Back to the board", nextWeek: "Next week →", finalBtn: "🏁 See your final report",
    badges: {},
    seasonEyebrow: (n) => `Season report · ${n} weeks`,
    avgScore: (s) => `Average success score <b>${s}/100</b>.`,
    fYourNet: "Your net value", fOver: (n) => `over ${n} weeks`,
    fModels: "Following the models", fModelsS: "each week's chosen model's plan",
    fHindsight: "Perfect hindsight", fHindsightS: "the ceiling",
    fWastedLost: "Wasted · lost", fWastedLostS: "false positives · false negatives",
    fiveThings: "Five things a Chief Retention Officer now knows",
    takeaways: [
      "📈 <b>A churn probability is a ranking, not a verdict.</b> Better models (higher ROC-AUC) put real churners nearer the top, so the same budget reaches more of them.",
      "🎚️ <b>The threshold is a business decision.</b> 50% is a convention; with ~20% churners, a lower, cross-validated threshold catches far more leavers.",
      "⚖️ <b>Precision protects the budget, Recall protects revenue.</b> You cannot maximise both; choose the trade-off that makes money.",
      "🎲 <b>Five customers are noise, two thousand are evidence.</b> Judge models on cross-validation and at scale, not on one lucky week.",
      "🧪 <b>Never tune on the test set.</b> These outcomes come from held-out customers the models never saw, which is why they are believable.",
    ],
    playAgain: "🔁 Play again",
    chapters: (x) => [
      { title: "Chapter 1 · Your first day",
        text: `<p class="lead">Congratulations: you are the bank's new <b>Chief Retention Officer</b>. Last year <b>${x.base}</b> of customers walked out of the door, taking their balances and fees with them.</p>
               <p class="lead">Every week you get <b>${x.budget}</b> and five customers on your desk. Spend it on calls, fee waivers or VIP packages to keep the ones who are about to leave. But every dollar spent on a customer who was staying anyway is wasted.</p>`,
        visual: `<div class="compare">${x.incentives.map((i) => `<div class="stat"><div class="k">${i.icon} ${i.name}</div><div class="v">${i.cost}</div><div class="s">${i.success} chance a would-be churner stays · ${i.blurb}</div></div>`).join("")}<div class="stat"><div class="k">Your budget</div><div class="v">${x.budget}</div><div class="s">per week, for ${x.weeks} weeks</div></div></div>` },
      { title: "Chapter 2 · Your crystal ball: churn risk probability",
        text: `<p class="lead">You cannot ask customers whether they plan to leave. But the data science team trained <b>${x.nModels} machine learning models</b> on 8,000 past customers. For anyone on your desk, a model returns a <b>churn risk probability</b>: P(customer leaves).</p>
               <p class="lead">Models are not equally good. Their ranking quality is measured by <b>ROC-AUC</b> (0.5 = coin flip, 1.0 = perfect): ${x.modelList}. You can switch models live during the game and see your results change.</p>`,
        visual: `<div class="eyebrow">Example: a customer scored 62%</div><div class="mini-bar"><div class="mini-fill" style="width:62%"></div></div>
                 <div class="compare"><div class="stat"><div class="k">Best ranker</div><div class="v">${x.best}</div><div class="s">CV ROC-AUC ${x.bestAuc}</div></div>
                 <div class="stat"><div class="k">Linear baseline</div><div class="v">Log. Regression</div><div class="s">CV ROC-AUC ${x.lrAuc}</div></div></div>` },
      { title: "Chapter 3 · Your rule: the decision threshold",
        text: `<p class="lead">A probability is not a decision. You need a rule: <b>flag a customer when risk ≥ threshold</b>. The textbook default is 50%. But with only one churner in five, ${x.best} rarely gets that sure, and a 50% rule misses many leavers.</p>
               <p class="lead">Using cross-validation on training data only, the team found that <b>${x.opt}</b> balances catches and false alarms best for ${x.best}. On all ${x.n} test customers:</p>`,
        visual: `${x.thresholdBar}
                 <div class="compare"><div class="stat"><div class="k">Threshold 50%</div><div class="v">${x.r50}</div><div class="s">of churners caught (Recall) · ${x.p50} of flags correct (Precision)</div></div>
                 <div class="stat"><div class="k">Threshold ${x.opt}</div><div class="v">${x.rOpt}</div><div class="s">of churners caught (Recall) · ${x.pOpt} of flags correct (Precision)</div></div></div>` },
      { title: "Chapter 4 · The trade-off you will live with",
        text: `<p class="lead"><b>Precision</b> asks: of the customers I spent money on, how many were really leaving? Low precision = <b>wasted budget</b> on loyal customers (false positives).</p>
               <p class="lead"><b>Recall</b> asks: of the customers who left, how many did I reach? Low recall = <b>lost revenue</b> from churners nobody called (false negatives).</p>
               <p class="lead">Lowering the threshold trades one for the other. Your job: find the balance that makes money. Each week ends with a debrief that shows what the data really said.</p>`,
        visual: `<div class="compare"><div class="stat"><div class="k">⚠️ False positive</div><div class="v" style="color:var(--amber)">Wasted</div><div class="s">You paid a loyal customer who was staying anyway.</div></div>
                 <div class="stat"><div class="k">❌ False negative</div><div class="v" style="color:var(--red)">Lost</div><div class="s">A churner left and nobody tried to keep them.</div></div>
                 <div class="stat"><div class="k">✅ True positive</div><div class="v" style="color:var(--green)">Saved</div><div class="s">You reached a churner; the incentive may keep them.</div></div>
                 <div class="stat"><div class="k">👍 True negative</div><div class="v" style="color:var(--blue)">Fine</div><div class="s">A loyal customer, correctly left alone.</div></div></div>` },
    ],
  };

  const he = {
    dir: "rtl",
    eyebrow: "סיפור החלטות בלמידת מכונה",
    title: "🏦 מנהלי השימור",
    back: "→ הקודם", skip: "דילוג על הסיפור", next: "הבא ←", startWeek1: "מתחילים את שבוע 1 ←",
    weekOf: (w, n) => `שבוע ${w} מתוך ${n}`,
    deskTitle: "חמישה לקוחות על השולחן שלכם",
    storyBtn: "📖 הסיפור",
    hudBudget: "תקציב שנותר", hudCommitted: "הוקצה",
    targeted: (n) => (n === 1 ? "לקוח אחד בטיפול" : `${n} לקוחות בטיפול`),
    hudExpSave: "חיסכון צפוי לפי המודל", hudExpSaveS: (m) => `אם הסיכונים של ${m} נכונים`,
    hudExpRoi: "ROI צפוי", hudRoiS: "(חיסכון − הוצאה) / הוצאה",
    hudSeason: "סיכום העונה", hudSeasonS: "ערך נקי בשבועות שנחשפו",
    hudRealSave: "✅ חיסכון בפועל", hudRealSaveS: (w, l) => `בוזבזו ${w} · אבדו ${l}`,
    hudRealRoi: "📈 ROI בפועל", netShort: (n) => `נקי ${n}`, na: "לא רלוונטי",
    crystal: "🧠 כדור הבדולח שלכם: בחרו מודל",
    thresholdTitle: "🎚️ סף החלטה",
    useOpt: "הסף האופטימלי של המודל", default50: "ברירת מחדל 50%", autoIncLabel: "תמריץ ללקוחות המסומנים",
    targetAll: "🎯 פנו לכל המסומנים", clear: "ניקוי",
    thresholdHint: "לקוחות עם סיכון ≥ הסף <b>מסומנים</b> (זוהרים באדום). הורידו אותו כדי לתפוס יותר נוטשים (Recall ↑), העלו אותו כדי לבזבז פחות על לקוחות נאמנים (Precision ↑).",
    scaleTitle: (n) => `🔭 הכלל הזה בקנה מידה: כל ${n} לקוחות הבדיקה`,
    readyText: "מוכנים? התוצאות הן התוצאות <i>האמיתיות</i> של הלקוחות האלה, שהוחזקו בצד.",
    revealBtn: "🎲 חשפו מה קרה", openDebrief: "📋 פתיחת התחקיר",
    revealHint: "לחצו על כרטיס כדי לשוחח עם הלקוח, ואז בחרו תמריץ בכפתורים.",
    revealedHint: "התוצאות נחשפו. פתחו את התחקיר ללקחי השבוע.",
    riskHigh: "😠 סיכון גבוה", riskWavering: "😐 בהתלבטות", riskLoyal: "😊 סיכון נמוך",
    chipMeta: (auc, opt) => `ROC-AUC ${auc} · סף אופטימלי ${opt}`,
    modelHint: (m, p) => `${m} מדרג נוטש אקראי מעל לקוח נאמן אקראי ב-<b>${p}</b> מהמקרים (באימות צולב). ⭐ = המודל שנבחר באימות צולב.`,
    optimumOf: (m, p) => `הסף האופטימלי של ${m}: ${p}`,
    churnRisk: "סיכון נטישה",
    geo: { France: "צרפת", Germany: "גרמניה", Spain: "ספרד" },
    facts: (c, geo, balance) => `גיל ${c.age} · ${geo} · ${c.products} מוצרים · ${c.active ? "חשבון פעיל" : "חשבון לא פעיל"} · ותק ${c.tenure} שנים<br>יתרה ${balance}`,
    valueAtStake: "ערך בסיכון", noAction: "ללא פעולה", none: "ללא",
    actTitle: (name, cost, succ) => `${name}: ${cost}, ${succ} הצלחה`,
    overBudget: "חריגה מהתקציב! פנו כסף ממקום אחר קודם.",
    scaleIncName: (icon, name) => `${icon} ${name} לכל לקוח מסומן`,
    sFlagged: "מסומנים", sFlaggedS: (tp, fp) => `${tp} נוטשים + ${fp} נאמנים`,
    sRecall: "Recall", sRecallS: (tp, all) => `${tp} מתוך ${all} נוטשים נתפסו`,
    sPrecision: "Precision", sPrecisionS: (tp, f) => `${tp} מתוך ${f} סימונים נכונים`,
    sWasted: "בזבוז (FP)", sWastedS: (fp) => `שולם ל-${fp} לקוחות נאמנים`,
    sMissed: "פספוסים (FN)", sMissedS: "נוטשים שאיש לא פנה אליהם",
    sNet: "ערך נקי", sNetS: (sav, sp) => `חיסכון ${sav} − הוצאה ${sp}`,
    scaleHint: (m, t) => `ספירות אמיתיות מקבוצת הבדיקה עבור ${m} בסף ${t}. הזיזו את המחוון או החליפו מודל, וצפו ב-Precision וב-Recall מחליפים מקומות.`,
    resTP: (x) => `🚪 נטישה · ✅ הגעתם בזמן: חיסכון צפוי ${x}`,
    resFN: (x) => `🚪 נטישה · ❌ פספוס: אבדו ${x}`,
    resFP: (x) => `🏠 ללא נטישה · ⚠️ בוזבזו ${x}`,
    resTN: "🏠 ללא נטישה · 👍 נכון לא לפנות",
    list: (names) => names.join(", "),
    lThreshold: (t, names, scores, maxT, many) => `🎚️ <b>הסף שלכם (${t}) היה מחמיר מדי עבור: ${names}</b>. הציונים היו ${scores}, והנטישה קרתה. סף של ${maxT} או פחות היה מסמן ${many ? "את כולם" : "גם את הלקוח הזה"}. זה ה-Recall: נוטשים חומקים כשהרף גבוה מדי.`,
    lFalseFlags: (names) => `🎯 <b>סומנו כבסיכון אך לא עזבו: ${names}.</b> זה מחיר ה-Precision של הסף שלכם: אזעקות על לקוחות נאמנים.`,
    lWasted: (x, names) => `⚠️ <b>בוזבזו ${x}</b> על ${names}: חיוביים שגויים, תמריצים ללקוחות שהיו נשארים בכל מקרה.`,
    lLost: (x, names) => `❌ <b>אבדו ${x}</b>: ${names} עזבו בלי שום ניסיון שימור. אלה שליליים שגויים.`,
    lModelWeek: (top, diff, cur) => `🧠 <b>בחירת המודל השבוע:</b> מעקב אחר הסיכונים של ${top} היה מניב ${diff} יותר מאשר מעקב אחר ${cur}.`,
    lModelPaid: (cur, net) => `🧠 <b>בחירת המודל השתלמה השבוע:</b> אף תוכנית של מודל אחר לא ניצחה את זו של ${cur} (${net}).`,
    lScale: (inc, n, bestNet, best, worstNet, worst) => `📊 <b>בקנה מידה, המודל קובע:</b> קמפיין "${inc}" על ${n} לקוחות מניב ${bestNet} עם ${best} אבל ${worstNet} עם ${worst}, כל אחד בסף האופטימלי שלו. שבוע אחד של חמישה לקוחות הוא רעש; אלפי החלטות הן ראיה.`,
    debEyebrow: (w, m, t) => `תחקיר שבוע ${w} · מודל ${m} · סף ${t}`,
    victory: "ניצחון", defeat: "הפסד",
    scoreLine: (s) => `ציון הצלחה ${s}/100 = הערך הנקי שלכם כחלק מהטוב ביותר שאפשר היה להשיג השבוע.`,
    kSavings: "✅ חיסכון בפועל", kSavingsS: "נוטשים שהגעתם אליהם × הצלחה × ערך",
    kWasted: "⚠️ תקציב מבוזבז", kWastedS: "חיוביים שגויים",
    kLost: "❌ הכנסות שאבדו", kLostS: "נוטשים שפוספסו + תמריצים שנכשלו",
    kRoi: "📈 ROI נקי",
    weekTeaches: "מה השבוע מלמד", youVs: "אתם מול נקודות הייחוס (השבוע)",
    you: "אתם", planOf: (m) => `התוכנית של ${m}`, hindsight: "ידיעה מושלמת בדיעבד",
    everyModel: "התוכנית של כל מודל, תוצאות אמיתיות",
    foot: "תוכנית של מודל ממקסמת ערך צפוי לפי ציוני הסיכון שלו, בתוך התקציב. הידיעה בדיעבד יודעת מי עזב.",
    backBoard: "חזרה ללוח", nextWeek: "השבוע הבא ←", finalBtn: "🏁 לדוח הסיכום",
    badges: {
      "Retention Champion": ["אלופי השימור", "השגתם את רוב הערך שמנהל עם ידיעה מושלמת היה יכול להציל."],
      "Savvy Strategist": ["אסטרטגים נבונים", "החלטות מבוססות נתונים, עם תשואה בריאה."],
      "Promising Manager": ["מנהלים מבטיחים", "כמה החלטות טובות, אבל נשאר כסף על השולחן או שהוצא על לקוחות נאמנים."],
      "Cautious Starter": ["צעדים ראשונים זהירים", "לא הפסדתם כסף, אבל רוב הערך שאפשר היה להציל חמק."],
      "Budget Burner": ["שורפי תקציב", "התקציב גרם יותר נזק מתועלת. בדקו את ציוני הסיכון ונסו שוב."],
    },
    seasonEyebrow: (n) => `דוח עונה · ${n} שבועות`,
    avgScore: (s) => `ציון הצלחה ממוצע <b>${s}/100</b>.`,
    fYourNet: "הערך הנקי שלכם", fOver: (n) => `לאורך ${n} שבועות`,
    fModels: "מעקב אחר המודלים", fModelsS: "התוכנית של המודל שנבחר בכל שבוע",
    fHindsight: "ידיעה מושלמת בדיעבד", fHindsightS: "התקרה",
    fWastedLost: "בוזבז · אבד", fWastedLostS: "חיוביים שגויים · שליליים שגויים",
    fiveThings: "חמישה דברים שמנהלי שימור יודעים עכשיו",
    takeaways: [
      "📈 <b>הסתברות נטישה היא דירוג, לא פסק דין.</b> מודלים טובים יותר (ROC-AUC גבוה יותר) מציבים נוטשים אמיתיים קרוב יותר לראש הרשימה, כך שאותו תקציב מגיע ליותר מהם.",
      "🎚️ <b>הסף הוא החלטה עסקית.</b> 50% הוא מוסכמה; כשיש כ-20% נוטשים, סף נמוך יותר שנקבע באימות צולב תופס הרבה יותר עוזבים.",
      "⚖️ <b>Precision שומר על התקציב, Recall שומר על ההכנסות.</b> אי אפשר למקסם את שניהם; בחרו את האיזון שמרוויח כסף.",
      "🎲 <b>חמישה לקוחות הם רעש, אלפיים הם ראיה.</b> שפטו מודלים לפי אימות צולב ובקנה מידה, לא לפי שבוע אחד של מזל.",
      "🧪 <b>לעולם לא מכווננים על קבוצת הבדיקה.</b> התוצאות האלה מגיעות מלקוחות שהמודלים מעולם לא ראו, ולכן אפשר להאמין להן.",
    ],
    playAgain: "🔁 משחק חוזר",
    chapters: (x) => [
      { title: "פרק 1 · היום הראשון שלכם",
        text: `<p class="lead">ברכות: מיניתם לראשי מחלקת <b>השימור</b> החדשים של הבנק. בשנה שעברה <b>${x.base}</b> מהלקוחות יצאו מהדלת, ולקחו איתם את היתרות והעמלות שלהם.</p>
               <p class="lead">בכל שבוע תקבלו <b>${x.budget}</b> וחמישה לקוחות על השולחן. השקיעו בשיחות, בוויתור על עמלות או בחבילות VIP כדי לשמר את מי שעומדים לעזוב. אבל כל דולר שמושקע בלקוח שהיה נשאר בכל מקרה הוא דולר מבוזבז.</p>`,
        visual: `<div class="compare">${x.incentives.map((i) => `<div class="stat"><div class="k">${i.icon} ${i.name}</div><div class="v">${i.cost}</div><div class="s">${i.success} סיכוי שלקוח שעמד לנטוש יישאר · ${i.blurb}</div></div>`).join("")}<div class="stat"><div class="k">התקציב שלכם</div><div class="v">${x.budget}</div><div class="s">לשבוע, במשך ${x.weeks} שבועות</div></div></div>` },
      { title: "פרק 2 · כדור הבדולח: הסתברות סיכון הנטישה",
        text: `<p class="lead">אי אפשר לשאול לקוחות אם הם מתכננים לעזוב. אבל צוות מדעי הנתונים אימן <b>${x.nModels} מודלים של למידת מכונה</b> על 8,000 לקוחות עבר. לכל לקוח על השולחן, מודל מחזיר <b>הסתברות סיכון נטישה</b>: P(הלקוח עוזב).</p>
               <p class="lead">המודלים אינם טובים באותה מידה. איכות הדירוג שלהם נמדדת ב-<b>ROC-AUC</b> (0.5 = הטלת מטבע, 1.0 = מושלם): ${x.modelList}. אפשר להחליף מודלים בזמן אמת במהלך המשחק ולראות את התוצאות משתנות.</p>`,
        visual: `<div class="eyebrow">דוגמה: לקוח עם ציון 62%</div><div class="mini-bar"><div class="mini-fill" style="width:62%"></div></div>
                 <div class="compare"><div class="stat"><div class="k">המדרג הטוב ביותר</div><div class="v">${x.best}</div><div class="s">ROC-AUC ב-CV ${x.bestAuc}</div></div>
                 <div class="stat"><div class="k">בסיס לינארי</div><div class="v">Log. Regression</div><div class="s">ROC-AUC ב-CV ${x.lrAuc}</div></div></div>` },
      { title: "פרק 3 · הכלל שלכם: סף ההחלטה",
        text: `<p class="lead">הסתברות היא לא החלטה. צריך כלל: <b>מסמנים לקוח כאשר הסיכון ≥ הסף</b>. ברירת המחדל של ספרי הלימוד היא 50%. אבל כשרק אחד מכל חמישה לקוחות נוטש, ${x.best} כמעט אף פעם לא בטוח עד כדי כך, וכלל של 50% מפספס עוזבים רבים.</p>
               <p class="lead">בעזרת אימות צולב על נתוני האימון בלבד, הצוות מצא שסף של <b>${x.opt}</b> מאזן הכי טוב בין תפיסות לאזעקות שווא עבור ${x.best}. על כל ${x.n} לקוחות הבדיקה:</p>`,
        visual: `${x.thresholdBar}
                 <div class="compare"><div class="stat"><div class="k">סף 50%</div><div class="v">${x.r50}</div><div class="s">מהנוטשים נתפסו (Recall) · ${x.p50} מהסימונים נכונים (Precision)</div></div>
                 <div class="stat"><div class="k">סף ${x.opt}</div><div class="v">${x.rOpt}</div><div class="s">מהנוטשים נתפסו (Recall) · ${x.pOpt} מהסימונים נכונים (Precision)</div></div></div>` },
      { title: "פרק 4 · האיזון שתחיו איתו",
        text: `<p class="lead"><b>Precision</b> שואל: מבין הלקוחות שהשקעתי בהם כסף, כמה באמת עמדו לעזוב? Precision נמוך = <b>תקציב מבוזבז</b> על לקוחות נאמנים (חיוביים שגויים).</p>
               <p class="lead"><b>Recall</b> שואל: מבין הלקוחות שעזבו, לכמה הגעתי? Recall נמוך = <b>הכנסות שאבדו</b> מנוטשים שאיש לא התקשר אליהם (שליליים שגויים).</p>
               <p class="lead">הורדת הסף מחליפה אחד בשני. התפקיד שלכם: למצוא את האיזון שמרוויח כסף. כל שבוע מסתיים בתחקיר שמראה מה הנתונים באמת אמרו.</p>`,
        visual: `<div class="compare"><div class="stat"><div class="k">⚠️ חיובי שגוי</div><div class="v" style="color:var(--amber)">בזבוז</div><div class="s">שילמתם ללקוח נאמן שהיה נשאר בכל מקרה.</div></div>
                 <div class="stat"><div class="k">❌ שלילי שגוי</div><div class="v" style="color:var(--red)">הפסד</div><div class="s">לקוח עזב ואיש לא ניסה לשמר אותו.</div></div>
                 <div class="stat"><div class="k">✅ חיובי אמיתי</div><div class="v" style="color:var(--green)">הצלה</div><div class="s">הגעתם לנוטש; התמריץ עשוי לשמר אותו.</div></div>
                 <div class="stat"><div class="k">👍 שלילי אמיתי</div><div class="v" style="color:var(--blue)">תקין</div><div class="s">לקוח נאמן, שבצדק לא פניתם אליו.</div></div></div>` },
    ],
  };

  const I18N = { en, he };
  if (typeof module !== "undefined" && module.exports) module.exports = I18N;
  else root.I18N = I18N;
})(typeof window !== "undefined" ? window : this);
