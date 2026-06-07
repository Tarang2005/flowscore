# FlowScore — Project Validation & Drift Detection Guide
**Use this to verify you're building the RIGHT thing, not just building SOMETHING**

---

## HOW TO USE THIS DOCUMENT

At the END of each week (or each block), copy the relevant section below and:

1. Paste it into Claude.ai or Claude Code as a prompt
2. Paste YOUR actual code/output below the prompt
3. Let Claude verify alignment with FLOWSCORE_PROJECT_CONTEXT.md
4. Fix any drift before moving to the next block

---

## WEEK 1 VALIDATION PROMPT
**Use this after you complete blocks 1.1–1.4 (Data + Model)**

```
You are a project quality auditor. I'm building FlowScore (gig worker credit scoring) 
for a hackathon. Below is my actual code/output from Week 1.

VALIDATE against the FlowScore context:
https://LINK_TO_YOUR_CONTEXT_DOC (or paste context inline)

Check these SPECIFIC REQUIREMENTS:

ARCHITECTURE:
✓ Did I create exactly 23 features (not more, not less)?
✓ Are they the RIGHT 23 (from section 3 of context)?
✓ Do my feature names match the context spec?

DATA:
✓ Am I using Home Credit dataset (the public Kaggle one)?
✓ Is my target variable correct (default/no-default)?
✓ Did I handle missing values (median/mode, drop >50%)?
✓ Is my train/test split 70/30?

MODEL:
✓ Did I use XGBoost (not a different algorithm)?
✓ Are my hyperparameters close to spec? (n_estimators=200, max_depth=6, learning_rate=0.05, etc.)
✓ Did I use scale_pos_weight=10 (class imbalance handling)?
✓ Is my AUC reported? (target: ≥85%, acceptable: ≥80%)
✓ Did I create SHAP explainer (TreeExplainer, not KernelExplainer)?

OUTPUTS:
✓ Did I save 4 pickle files? (model.pkl, scaler.pkl, explainer.pkl, feature_names.pkl)
✓ Can I load these files without errors?
✓ Did I create a feature importance plot?

SCRIPT QUALITY:
✓ Is my code well-commented?
✓ Did I use logging (not just print statements)?
✓ Does the script have error handling?

TESTING:
✓ Can I run the script end-to-end without errors?
✓ Do the metrics make sense (not 99% AUC, not 50%)?

DRIFT DETECTION:
⚠️  Answer these to catch common mistakes:
- Did I accidentally use test data to train the model? (common mistake)
- Did I scale features BEFORE or AFTER train/test split? (should be AFTER)
- Did I use all 122 Home Credit features or select to 23? (must select to 23)
- Is my SHAP explainer getting reasonable explanations? (test with 1 sample)

OVERALL: Rate alignment as RED, YELLOW, or GREEN.
- GREEN: Matches context, ready for Week 2
- YELLOW: Minor deviations (acceptable with explanation)
- RED: Major divergence (fix before moving forward)

If RED or YELLOW, explain what diverged and why.
```

**After Claude responds, answer this yourself:**
- [ ] Did Claude give GREEN status?
- [ ] If YELLOW, do I understand and accept the deviation?
- [ ] Can I proceed to Week 2 with confidence?

---

## WEEK 2 VALIDATION PROMPT
**Use this after you complete blocks 2.1–2.4 (FastAPI Backend)**

```
You are a backend quality auditor. I'm building FlowScore's FastAPI backend.
Below is my code.

VALIDATE against the FlowScore context (section 5: API Specifications).

API ENDPOINTS - REQUIRED:
✓ POST /score — Does it accept BorrowerProfile and return ScoreResponse?
✓ GET /borrower/{id} — Does it return full profile + score history?
✓ POST /ingest — Does it accept mock webhook and return updated score?
✓ GET /health — Does it return {"status": "healthy"}?
✓ GET /demo/personas — Does it return list of 3 personas?

SCORE RESPONSE STRUCTURE - Must match exactly:
✓ flowscore (int, 0-850)
✓ default_probability (float, 0-1)
✓ risk_category ("low"/"medium"/"high")
✓ shap_explanation (dict with top_positive_factors, top_negative_factors)
✓ coaching_tips (list of 3 strings)
✓ model_metadata (version, timestamp, feature_count)

BORROWER PROFILE INPUT - Must accept:
✓ income_data (with platforms array)
✓ spending_data (with categories)
✓ credit_profile
✓ calculated_features

FEATURE ENGINEERING:
✓ Do I featurize the borrower profile into exactly 23 features?
✓ Are they in the same ORDER as the trained model?
✓ Do I scale using the scaler.pkl from Week 1?

ML MODEL INTEGRATION:
✓ Did I load model.pkl, scaler.pkl, explainer.pkl correctly?
✓ Can I call model.predict() on a sample?
✓ Can I generate SHAP values without errors?
✓ Is inference time <200ms per request? (test with time.time())

SHAP EXPLANATIONS:
✓ Are the top 5 factors correctly extracted?
✓ Do the factor names match my 23 features?
✓ Do the contribution values make sense (-100 to +100 range)?

COACHING TIPS GENERATION:
✓ Did I create 3 tips per prediction?
✓ Are they based on actual SHAP factors (not hardcoded)?
✓ Do they make sense? (e.g., "income declining → grow income")
✓ Do they include estimated score impact? (+X points)

CODE QUALITY:
✓ Are there Pydantic schemas for all request/response types?
✓ Is there error handling (400s for bad input, 500s for server errors)?
✓ Do all endpoints have docstrings?
✓ Is there request logging?

TESTING - Critical:
✓ Does GET /health return 200 immediately?
✓ Does POST /score with Priya's persona return a valid response?
✓ Is Priya's score in the 600-650 range (expected: 625)?
✓ Does POST /score with Arjun return 750+ (expected: 760)?
✓ Does POST /ingest update the score correctly?

PERSONAS DATA:
✓ Did I create personas.json with Priya, Arjun, Meera?
✓ Do they have the full BorrowerProfile structure?
✓ Can I load and serve them via GET /demo/personas?

DEPLOYMENT READINESS:
✓ Does the app run with: uvicorn main:app --reload?
✓ Are all imports available (no ImportError)?
✓ Are model files in the correct path (model/, relative or absolute)?

DRIFT DETECTION:
⚠️  Common Week 2 mistakes:
- Did I accidentally change the feature order from Week 1? (breaking)
- Did I forget to scale features before prediction? (breaking)
- Did I hardcode coaching tips instead of deriving from SHAP? (wrong approach)
- Did I make the API too complicated? (keep it simple: 1 POST, 1 GET for borrower, 1 for personas)
- Did I spend time on database/persistence? (skip this for MVP, use in-memory)

OVERALL: Rate alignment as RED, YELLOW, or GREEN.
```

**After Claude responds:**
- [ ] Did Claude give GREEN?
- [ ] Can I test the API locally with curl/Postman?
- [ ] Do the 3 personas return the expected score ranges?

---

## WEEK 3 VALIDATION PROMPT
**Use this after you complete blocks 3.1–3.4 (React Frontend)**

```
You are a frontend quality auditor. I'm building FlowScore's React dashboard.
Below is my code.

VALIDATE against the FlowScore context (section 6: Frontend Specifications).

COMPONENTS - Required 5:
✓ ScoreGauge.jsx
  - Circular gauge, 0-850 scale
  - Color: Green (700+), Yellow (600-699), Red (<600)
  - Shows current score + 1-month change
  
✓ IncomeChart.jsx
  - Line chart, 6 months
  - X: month labels, Y: income (₹)
  - Tooltip on hover
  
✓ SHAPBreakdown.jsx
  - Horizontal bar chart
  - Green bars (top 5 positive), Red bars (top 3 negative)
  - Feature names + contribution values
  
✓ CoachingTips.jsx
  - 3 cards with icon, text, score impact
  - Color-coded (green/yellow/red)
  
✓ PersonaSelector.jsx
  - Dropdown: Priya, Arjun, Meera
  - On select: load borrower from API

PAGES - Required 2:
✓ Dashboard.jsx
  - Layout: ScoreGauge (1/3) + Rest (2/3)
  - IncomeChart, SHAPBreakdown, CoachingTips stacked
  - PersonaSelector at top
  - Loading spinner while fetching
  - Error fallback UI
  
✓ LenderView.jsx
  - Show raw API response (ScoreResponse JSON)
  - Display in table/card format (not raw JSON)
  - Copy-to-clipboard button

API INTEGRATION:
✓ Did I create src/services/api.js?
✓ Does it have fetchScore(), fetchBorrower(), fetchPersonas()?
✓ Are API calls using VITE_API_URL from .env?
✓ Is error handling in place (catch blocks)?
✓ Do I handle loading states (useState)?

DESIGN SYSTEM:
✓ Are all components using Tailwind CSS?
✓ Is the color palette correct?
  - Primary: #185FA5 (blue)
  - Success: #3B6D11 (green)
  - Warning: #854f0b (amber)
  - Danger: #A32D2D (red)
✓ Is spacing 8px-based (8, 16, 24, 32px)?
✓ Do buttons have hover states?
✓ Is the layout responsive (mobile-friendly)?

DATA FLOW - Critical:
✓ Does clicking persona selector trigger API call?
✓ Does API response flow to all components?
✓ Do charts update when data changes?
✓ Is the income array correctly formatted for Recharts?
✓ Is the SHAP breakdown correctly mapped from response?

CHARTS (Recharts):
✓ IncomeChart: Does LineChart render with correct data?
✓ SHAPBreakdown: Does BarChart show positive (green) and negative (red)?
✓ ScoreGauge: Does the gauge render correctly? (test with PieChart if needed)

TESTING - User Perspective:
✓ Can I load the app (npm run dev)?
✓ Do I see the persona dropdown?
✓ Can I select Priya without errors?
✓ Does the API call succeed (check Network tab in DevTools)?
✓ Does Priya's score display as ~625?
✓ Do charts render with data?
✓ Do coaching tips make sense for Priya?
✓ Can I switch to Arjun and see different score (~750+)?
✓ Can I switch to LenderView and see JSON?
✓ Is everything readable on mobile (iPhone SE size)?

CODE QUALITY:
✓ Are all components using functional components + hooks?
✓ Is state management clean (not nested useState)?
✓ Are API calls in useEffect (not on render)?
✓ Are there PropTypes or TypeScript? (optional but good)
✓ Is code well-commented?

STYLING:
✓ Does the UI look professional? (not broken, aligned, clean)
✓ Are colors consistent throughout?
✓ Is there good spacing/whitespace?
✓ Do components align nicely (no overlap)?

DRIFT DETECTION:
⚠️  Common Week 3 mistakes:
- Did I add a database/backend persistence? (skip for MVP)
- Did I create 10 components instead of 5? (overcomplicated)
- Did I forget .env file with VITE_API_URL? (breaking)
- Did I hardcode API URL instead of using .env? (won't work in production)
- Did I use localStorage to cache scores? (not needed, just fetch fresh)
- Did I spend time on animations? (skip, focus on functionality)

OVERALL: Rate alignment as RED, YELLOW, or GREEN.
```

**After Claude responds:**
- [ ] Did Claude give GREEN?
- [ ] Test with all 3 personas — do they load without errors?
- [ ] Does it look "hackathon-winning" (not ugly)?

---

## WEEK 4 VALIDATION PROMPT
**Use this after you complete blocks 4.1–4.5 (Deploy + Polish)**

```
You are a launch quality auditor. I'm deploying FlowScore.
Below is my deployment, docs, and demo prep.

DEPLOYMENT - Backend (Render):
✓ Is my FastAPI running on a public Render URL?
✓ Can I call /health and get 200 response?
✓ Can I POST /score from outside (not localhost)?
✓ Are there no CORS errors?
✓ Is model.pkl in the correct path on Render?

DEPLOYMENT - Frontend (Vercel):
✓ Is my React app deployed on Vercel?
✓ Can I visit the URL in Chrome, Firefox, Safari?
✓ Is VITE_API_URL pointing to the Render API (not localhost)?
✓ Do the persona selectors load and score correctly?
✓ Can I see API responses in DevTools Network tab?

DOCUMENTATION - README.md:
✓ Does README have:
  - Project title + tagline
  - Problem statement (70M gig workers, 73% rejected)
  - Solution overview
  - Live demo links (Vercel + Render URLs)
  - Architecture diagram
  - Tech stack (bulleted)
  - Getting started (setup instructions)
  - Model metrics (AUC, precision, recall)
  - Demo personas (Priya, Arjun, Meera)
  - Hackathon angle
  - Credits

✓ Are all links working?
✓ Are setup instructions clear enough for someone to clone and run?
✓ Does the README "sell" the project in first 30 seconds?

GITHUB REPO:
✓ Is code organized:
  - model/ folder
  - backend/ folder
  - frontend/ folder
  - README.md at root
  - .gitignore present
✓ Is repo public?
✓ Are there any secrets in the code? (API keys, etc. — should not be)
✓ Does README have GitHub repo link?

DEMO VIDEO (2 min):
✓ Did I record a 2-minute video?
✓ Does it show:
  - Live dashboard (not slides)
  - 2 different personas being scored
  - SHAP breakdown being explained
  - API response (lender view)
✓ Is the video clear (not pixelated)?
✓ Is the audio audible?
✓ Did I upload to YouTube (unlisted or public)?
✓ Does the video have a YouTube link in the README?

PITCH SCRIPT (2 min):
✓ Did I write a 2-minute pitch script?
✓ Does it follow this structure:
  - Hook (15s): Problem statement
  - Demonstration (60s): Live demo of the 3 personas
  - Market validation (20s): KarmaLife raised $8M, this is real
  - Call to action (5s): "Let's build the future of gig worker credit"
✓ Does it sound natural when read aloud (not robotic)?
✓ Can I deliver it in exactly 2 minutes?

TESTING CHECKLIST - Before Demo Day:
✓ Can I access both URLs (Vercel + Render) from any device?
✓ Do all 3 personas load and score correctly?
✓ Do charts render with real data?
✓ Are SHAP explanations sensible?
✓ Are coaching tips actionable?
✓ Is there no lag (APIs respond in <2 sec)?
✓ Did I test on mobile (iPhone size)?
✓ Did I test on different browsers?

SUBMISSION - Hackathon Form:
✓ GitHub repo URL (public)
✓ Live frontend URL (Vercel)
✓ Live backend URL (Render)
✓ Demo video URL (YouTube)
✓ README link
✓ Team member names
✓ Problem statement (copy from context)
✓ Solution summary (copy from context)
✓ Tech stack (copy from context)

DRIFT DETECTION - Final Check:
⚠️  Common Week 4 mistakes:
- Did I push secrets (API keys, passwords) to GitHub? (check!)
- Did I hardcode localhost URLs in production? (check!)
- Did I forget to update .env before deploying frontend? (check!)
- Did I leave console.log() statements everywhere? (clean them up)
- Did I test the live URLs from a friend's device? (do this!)
- Did I record the demo video or just plan to? (DO IT NOW)

FINAL CONFIDENCE CHECK:
Rate yourself (honestly):
- Technical quality: 1-10
- Design/UI: 1-10
- Pitch clarity: 1-10
- Hackathon fit: 1-10
- Overall readiness: 1-10

Average score <7 = Polish more. Average score ≥8 = Ready for demo day.

OVERALL: Rate alignment as RED, YELLOW, or GREEN.
```

**After Claude responds:**
- [ ] Did Claude give GREEN?
- [ ] Have I tested everything from a friend's device (not my own machine)?
- [ ] Am I confident in my pitch?

---

## CONTINUOUS ALIGNMENT CHECK
**Use this WEEKLY (not just at end of week)**

```
I'm on Week [X] of FlowScore. Quick alignment check:

BLOCK JUST COMPLETED: [block number]

STATUS CHECK:
1. Did I complete all the "test" requirements? (green checkboxes)
2. Do I have all the expected output files/components?
3. Have I diverged from FLOWSCORE_PROJECT_CONTEXT.md?

DIVERGENCE QUESTIONS (answer YES/NO):
- Am I still building exactly 23 features (not 15, not 50)? YES / NO
- Am I still using XGBoost (not LightGBM, not neural nets)? YES / NO
- Am I still targeting 85% AUC (not 99%, not 70%)? YES / NO
- Am I still using the 3 personas (Priya, Arjun, Meera)? YES / NO
- Am I still planning to deploy on Vercel + Render? YES / NO
- Am I still on pace to finish in 4 weeks at 20hrs/week? YES / NO

IF ANY "NO":
- Explain the deviation (why did I change?)
- Is the deviation better or worse for the project?
- Is it a blocker or acceptable?

CONFIDENCE:
On a scale 1-10, how confident am I that I'll ship a working demo by Week 4?

Current: [score]
```

---

## RED FLAGS — STOP AND REASSESS

If you see ANY of these, pause and validate with Claude:

1. **"I'm adding a database to persist user scores"**
   - Status: RED DRIFT
   - Reason: Context says "use in-memory for MVP"
   - Action: Remove persistence, use mock data

2. **"My model AUC is 72%, but I'm moving to Week 2"**
   - Status: YELLOW
   - Reason: Target is ≥85%, acceptable is ≥80%
   - Action: Spend 3 more days tuning, then proceed if ≥78%

3. **"I created 8 components instead of 5"**
   - Status: RED DRIFT
   - Reason: Over-engineering
   - Action: Consolidate 3 components, simplify

4. **"My FastAPI is running but it's taking 5 seconds per /score call"**
   - Status: YELLOW
   - Reason: Target is <200ms
   - Action: Profile code, optimize SHAP generation

5. **"I'm 3 weeks in and still on Week 2 work"**
   - Status: RED
   - Reason: Timeline is slipping
   - Action: Cut scope or increase hours. Skip optional features.

6. **"My frontend looks different from the figma/design"**
   - Status: YELLOW (if it looks better) / RED (if it looks worse)
   - Action: If worse, revert to Tailwind templates

7. **"I realized gig worker credit scoring is too complex, want to pivot"**
   - Status: RED DRIFT
   - Reason: You're 2+ weeks in
   - Action: Stick with it. Problems are always complex at start.

---

## HOW TO RESPOND TO VALIDATION

When Claude validates your work, they'll return one of 3 statuses:

### ✅ GREEN — You're on track
**Action**: Move to next block confidently. No changes needed.

### 🟡 YELLOW — Minor deviations, acceptable
**Action**: Understand the deviation. Decide if it's worth fixing before moving forward.

Example YELLOW responses:
- "Your model is 82% AUC, not 85%. This is acceptable for a hackathon demo. Proceed if you're OK with explaining this."
- "You created 6 components instead of 5. ScoreGauge and PersonaSelector could be combined. Not critical, but cleaner if combined."

### 🔴 RED — Major divergence, do not proceed
**Action**: Fix immediately before moving to next block. Don't stack technical debt.

Example RED responses:
- "Your features are in different order than training. This breaks predictions. Fix this now."
- "You're using 15 features instead of 23. Retrain with correct features before Week 2."

---

## TEMPLATE: Validation Submission Email/Message

When you're ready for Claude to validate:

```
WEEK [X] VALIDATION REQUEST

Block completed: [block number]
Date: [today's date]
Hours spent: [~5-10]

CODE/OUTPUT ATTACHED:
[paste your train.py / main.py / components folder / etc.]

SELF-ASSESSMENT:
- Did I complete all required steps? YES / NO
- Any deviations from context? [list them]
- AUC/metrics (if Week 1): [paste results]
- Persona scores (if Week 2+): Priya: XXX, Arjun: XXX, Meera: XXX

VALIDATION PROMPT:
[Paste the appropriate validation prompt above]
```

---

## SUCCESS METRICS

By end of Week 4, you should be able to answer YES to:

- [ ] Is my GitHub repo public and well-organized?
- [ ] Are both URLs (Vercel + Render) working from any device?
- [ ] Can I load 3 personas and see different scores?
- [ ] Are the scores in expected ranges (Priya: 625, Arjun: 750+, Meera: 550)?
- [ ] Do charts render correctly with real data?
- [ ] Are SHAP explanations showing and making sense?
- [ ] Do coaching tips align with SHAP factors?
- [ ] Is the UI clean and looks "polished"?
- [ ] Did I record and upload a 2-min demo video?
- [ ] Do I have a 2-min pitch script written?
- [ ] Have I tested from someone else's device (not just localhost)?
- [ ] Would I confidently demo this to 100 judges?

---

**Last Updated**: June 6, 2025
**Purpose**: Keep FlowScore on track, prevent drift, ensure hackathon readiness
**Use Every Week**: Not optional, not "nice to have"
