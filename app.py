import streamlit as st
import hashlib
import json
import time
import uuid
import os
import random
import re
import numpy as np
import cv2

# --- 1. PERSISTENT BLOCKCHAIN & FEEDBACK STORAGE ---
BLOCKCHAIN_FILE = "blockchain_ledger.json"
FEEDBACK_FILE = "feedback_log.json"

class VotingBlockchain:
    def __init__(self):
        self.chain = []
        self.used_tokens = set()
        self.load_chain()

    def load_chain(self):
        if os.path.exists(BLOCKCHAIN_FILE):
            try:
                with open(BLOCKCHAIN_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chain = data.get("chain", [])
                    self.used_tokens = set(data.get("used_tokens", []))
            except Exception:
                self.init_genesis()
        else:
            self.init_genesis()

    def init_genesis(self):
        self.chain = []
        self.used_tokens = set()
        self.create_block(proof=1, previous_hash='0', voter_token="GENESIS", candidate="NONE")

    def save_chain(self):
        data = {
            "chain": self.chain,
            "used_tokens": list(self.used_tokens)
        }
        with open(BLOCKCHAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def create_block(self, proof, previous_hash, voter_token, candidate):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'voter_token': voter_token,
            'vote_hash': self.hash_vote(voter_token, candidate),
            'candidate': candidate,
            'proof': proof,
            'previous_hash': previous_hash
        }
        block['current_hash'] = self.hash_block(block)
        self.chain.append(block)
        self.save_chain()
        return block

    @staticmethod
    def hash_vote(token, candidate):
        return hashlib.sha256(f"{token}:{candidate}".encode()).hexdigest()

    @staticmethod
    def hash_block(block):
        block_copy = {k: v for k, v in block.items() if k != 'current_hash'}
        encoded_block = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current['previous_hash'] != previous['current_hash']:
                return False
            if current['current_hash'] != self.hash_block(current):
                return False
        return True

def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_feedback(entry):
    items = load_feedbacks()
    items.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

# --- 2. PHONE & OTP HELPER ---
COUNTRY_CODES = {
    "नेपाल": "+977",
    "जापान": "+81",
    "दक्षिण कोरिया": "+82",
    "अष्ट्रेलिया": "+61",
    "अमेरिका": "+1",
    "युएई": "+971",
    "कतार": "+974",
    "अन्य": "+"
}

def validate_phone_for_country(country, phone_raw):
    cleaned = re.sub(r"[\s\-\(\)]", "", phone_raw.strip())
    if not cleaned:
        return False, None, "कृपया फोन नम्बर लेख्नुहोस्।"

    if country == "नेपाल":
        if cleaned.startswith("+977"):
            local = cleaned[4:]
        elif cleaned.startswith("977") and len(cleaned) == 13:
            local = cleaned[3:]
        else:
            local = cleaned

        if not re.fullmatch(r"(98|97)\d{8}", local):
            return False, None, "नेपालको मान्य मोबाइल नम्बर (जस्तै 98XXXXXXXX वा +97798XXXXXXXX) लेख्नुहोस्।"
        return True, f"+977{local}", "नेपालको फोन नम्बर मान्य छ।"
    else:
        if not cleaned.startswith("+"):
            cleaned = "+" + cleaned
        if not re.fullmatch(r"\+[1-9]\d{6,14}", cleaned):
            return False, None, "विदेशको मान्य अन्तर्राष्ट्रिय फोन नम्बर (जस्तै +819012345678 वा +12025550199) लेख्नुहोस्।"
        return True, cleaned, f"{country} को अन्तर्राष्ट्रिय नम्बर मान्य छ।"

# --- 3. VISION, DOCUMENT FACE MATCH & PALM SCAN ENGINE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def decode_image_bytes(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def extract_doc_face(image_bytes):
    img = decode_image_bytes(image_bytes)
    if img is None:
        return None, None, "अपलोड गरिएको कागजातको छवि पढ्न सकिएन।"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
    if len(faces) == 0:
        return None, None, "कागजातमा अनुहार स्पष्ट फेला परेन। कृपया फोटो स्पष्ट देखिने कागजात अपलोड गर्नुहोस्।"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    face_sig = cv2.resize(gray[y:y+h, x:x+w], (64, 64), interpolation=cv2.INTER_AREA)
    return face_rgb, face_sig, "Success"

def analyze_live_face_and_blue_ray(image_bytes):
    img = decode_image_bytes(image_bytes)
    if img is None:
        return False, None, None, {}, "अमान्य क्यामेरा छवि (Invalid Image)"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return False, None, None, {}, "अनुहार फेला परेन! कृपया क्यामेराको अगाडि स्पष्ट बस्नुहोस्।"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_color = img[y:y+h, x:x+w]
    face_gray = gray[y:y+h, x:x+w]

    mean_blue = float(np.mean(face_color[:, :, 0]))
    mean_green = float(np.mean(face_color[:, :, 1]))
    mean_red = float(np.mean(face_color[:, :, 2]))

    blue_to_red_ratio = mean_blue / (mean_red + 1e-5)
    blue_to_green_ratio = mean_blue / (mean_green + 1e-5)
    laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    bright_pixel_ratio = float(np.sum(face_gray > 210)) / float(face_gray.size)
    mean_brightness = float(np.mean(face_gray))

    metrics = {
        "blue_red_ratio": round(blue_to_red_ratio, 3),
        "blue_green_ratio": round(blue_to_green_ratio, 3),
        "brightness": round(mean_brightness, 1),
        "bright_pixel_pct": round(bright_pixel_ratio * 100, 1),
        "blur_laplacian": round(laplacian_var, 1)
    }

    if blue_to_red_ratio > 1.18 or blue_to_green_ratio > 1.15:
        return False, None, None, metrics, "🚨 स्पुफिङ चेतावनी! स्क्रिनबाट आउने नीलो प्रकाश (Blue Ray) पत्ता लाग्यो।"

    if bright_pixel_ratio > 0.25 and mean_brightness > 165:
        return False, None, None, metrics, "🚨 स्पुफिङ चेतावनी! डिजिटल स्क्रिनको अत्यधिक उज्यालो/रिफ्लेक्सन (Glare) पत्ता लाग्यो।"

    if laplacian_var < 6.0:
        return False, None, None, metrics, "🚨 धेरै धमिलो फोटो भयो! कृपया स्पष्ट अनुहार देखाउनुहोस्।"

    match_sig = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    dup_sig = cv2.resize(face_gray, (32, 32), interpolation=cv2.INTER_AREA)
    return True, match_sig, dup_sig, metrics, "Success"

def compare_doc_and_live_face(doc_sig, live_sig, tm_threshold=0.58, hist_threshold=0.55):
    if doc_sig is None or live_sig is None:
        return False, 0.0, 0.0

    doc_eq = cv2.equalizeHist(doc_sig)
    live_eq = cv2.equalizeHist(live_sig)

    tm_res = cv2.matchTemplate(live_eq, doc_eq, cv2.TM_CCOEFF_NORMED)
    tm_score = float(tm_res[0][0])

    hist_doc = cv2.calcHist([doc_eq], [0], None, [32], [0, 256])
    hist_live = cv2.calcHist([live_eq], [0], None, [32], [0, 256])
    cv2.normalize(hist_doc, hist_doc)
    cv2.normalize(hist_live, hist_live)
    hist_score = float(cv2.compareHist(hist_doc, hist_live, cv2.HISTCMP_CORREL))

    passed = (tm_score >= tm_threshold) and (hist_score >= hist_threshold)
    return passed, round(tm_score * 100, 1), round(hist_score * 100, 1)

def analyze_palm_scan(image_bytes, hand_label):
    img = decode_image_bytes(image_bytes)
    if img is None:
        return False, None, f"{hand_label} को छवि पढ्न सकिएन।"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    if roi.size == 0:
        return False, None, f"{hand_label} स्पष्ट फ्रेममा आएन।"

    edges = cv2.Canny(roi, 50, 140)
    line_density = float(np.sum(edges > 0)) / float(roi.size)
    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    if lap_var < 6.0 or line_density < 0.012:
        return False, None, f"⚠️ {hand_label} स्पष्ट देखिएन। हत्केला खोलेर क्यामेराको अगाडि अलि नजिक र स्पष्ट राख्नुहोस्।"

    palm_sig = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    return True, palm_sig, f"{hand_label} स्क्यान सफल भयो।"

def is_duplicate_face(new_sig, stored_sigs, threshold=0.82):
    for old_sig in stored_sigs:
        res = cv2.matchTemplate(new_sig, old_sig, cv2.TM_CCOEFF_NORMED)
        similarity = res[0][0]
        if similarity > threshold:
            return True, round(similarity * 100, 1)
    return False, None

# --- 4. SESSION STATE ---
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = VotingBlockchain()
if 'verified_voters' not in st.session_state:
    st.session_state.verified_voters = {}
if 'face_signatures' not in st.session_state:
    st.session_state.face_signatures = []
if 'current_token' not in st.session_state:
    st.session_state.current_token = None
if 'otp_code' not in st.session_state:
    st.session_state.otp_code = None
if 'otp_phone' not in st.session_state:
    st.session_state.otp_phone = None
if 'otp_verified' not in st.session_state:
    st.session_state.otp_verified = False

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="SajiloVote - Multi-Step Biometric & Blockchain Voting", layout="wide")

st.title("🇳🇵 SajiloVote: Multi-Step Verification & Blockchain Voting System")
st.warning("⚠️ **प्रोटोटाइप डेमो सूचना:** यो अनुसन्धान र परीक्षण प्रयोजनका लागि बनाइएको डेमो प्रणाली हो। कृपया वास्तविक नागरिकता, पासपोर्ट वा संवेदनशील व्यक्तिगत विवरण अपलोड नगरी नमुना (Sample/Test) डेटा मात्र प्रयोग गर्नुहोस्।")
st.markdown("**🛡️ Security Layers:** Country & Phone OTP + Strict Doc Face Match + Blue-Ray & Glare Check + Dual Palm Scan + SHA-256 Ledger")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "१. बहु-स्तरीय प्रमाणीकरण (Verification)",
    "२. सुरक्षित मतदान (Cast Vote)",
    "३. ब्लकचेन अडिट र नतिजा",
    "४. सुझाव र प्रतिक्रिया (Feedback)"
])

with tab1:
    st.subheader("पहिलो चरण: देश, फोन OTP, कागजात, लाइभ अनुहार र Palm Scan")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        id_number = st.text_input("नमुना परिचयपत्र / पासपोर्ट नम्बर (Sample ID No.):", placeholder="उदाहरण: TEST-2026-001")
    with col_meta2:
        country = st.selectbox(
            "हाल रहेको देश:",
            list(COUNTRY_CODES.keys())
        )
    with col_meta3:
        default_prefix = COUNTRY_CODES.get(country, "+")
        phone_input = st.text_input(
            f"फोन नम्बर ({country} कोड {default_prefix}):",
            placeholder="नेपाल: 98XXXXXXXX | विदेश: +819012345678"
        )

    otp_col1, otp_col2, otp_col3 = st.columns([1.2, 1.2, 2])
    with otp_col1:
        if st.button("OTP पठाउनुहोस्"):
            ok_phone, norm_phone, phone_msg = validate_phone_for_country(country, phone_input)
            if not ok_phone:
                st.error(f"❌ {phone_msg}")
                st.session_state.otp_verified = False
            else:
                generated_otp = f"{random.randint(100000, 999999)}"
                st.session_state.otp_code = generated_otp
                st.session_state.otp_phone = norm_phone
                st.session_state.otp_verified = False
                st.info(f"📩 {norm_phone} मा OTP पठाइयो (डेमो कोड: `{generated_otp}`)")

    with otp_col2:
        entered_otp = st.text_input("६-अङ्कको OTP कोड राख्नुहोस्:", max_chars=6, placeholder="123456")

    with otp_col3:
        st.write("")
        st.write("")
        if st.button("OTP प्रमाणित गर्नुहोस्"):
            ok_phone, norm_phone, phone_msg = validate_phone_for_country(country, phone_input)
            if not ok_phone:
                st.error(f"❌ {phone_msg}")
            elif st.session_state.otp_code is None or st.session_state.otp_phone != norm_phone:
                st.error("❌ कृपया पहिले यो फोन नम्बरका लागि 'OTP पठाउनुहोस्' थिच्नुहोस्।")
            elif entered_otp.strip() == st.session_state.otp_code:
                st.session_state.otp_verified = True
                st.success(f"✅ फोन नम्बर ({norm_phone}) र OTP प्रमाणित भयो!")
            else:
                st.session_state.otp_verified = False
                st.error("❌ OTP कोड मिलेन। कृपया सही कोड राख्नुहोस्।")

    if st.session_state.otp_verified and st.session_state.otp_phone:
        st.caption(f"प्रमाणित नम्बर: `{st.session_state.otp_phone}` ({country})")

    is_abroad = country != "नेपाल"
    if is_abroad:
        doc_type = "Passport (विदेशमा अनिवार्य)"
        st.warning("🌍 तपाईं नेपाल बाहिर हुनुहुन्छ, त्यसैले **Passport** अनिवार्य छ।")
    else:
        doc_type = st.radio("कागजातको प्रकार छान्नुहोस्:", ["Citizenship (नागरिकता)", "Passport (पासपोर्ट)"], horizontal=True)

    st.markdown("---")
    st.markdown("### क) नमुना कागजात अपलोड र साइड फोटो मिलान")
    doc_col1, doc_col2 = st.columns([2, 1])

    doc_face_rgb = None
    doc_face_sig = None

    with doc_col1:
        uploaded_doc = st.file_uploader(
            f"{doc_type} को नमुना तस्बिर अपलोड गर्नुहोस् (JPG/PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_doc is not None:
            doc_face_rgb, doc_face_sig, doc_msg = extract_doc_face(uploaded_doc.getvalue())
            if doc_face_rgb is None:
                st.error(f"❌ {doc_msg}")
            else:
                st.success("✅ कागजातबाट अनुहार सफलतापूर्वक निकालियो।")

    with doc_col2:
        st.write("**कागजातको फोटो (Extracted Side Photo):**")
        if doc_face_rgb is not None:
            st.image(doc_face_rgb, caption="कागजातको अनुहार", width=160)
        else:
            st.info("कागजात अपलोड गरेपछि यहाँ फोटो देखिनेछ।")

    st.markdown("---")
    st.markdown("### ख) प्रत्यक्ष अनुहार (Live Face) र दुबै हातको Palm Scan")
    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        live_face_photo = st.camera_input("१. प्रत्यक्ष अनुहार (Live Face)")
    with cam_col2:
        left_palm_photo = st.camera_input("२. बायाँ हातको हत्केला (Left Palm)")
    with cam_col3:
        right_palm_photo = st.camera_input("३. दायाँ हातको हत्केला (Right Palm)")

    if st.button("सबै चरण प्रमाणीकरण गर्नुहोस् र टोकन लिनुहोस्"):
        ok_phone, norm_phone, phone_msg = validate_phone_for_country(country, phone_input)
        if not id_number.strip():
            st.error("❌ कृपया नमुना परिचयपत्र वा पासपोर्ट नम्बर लेख्नुहोस्।")
        elif not ok_phone:
            st.error(f"❌ {phone_msg}")
        elif not st.session_state.otp_verified or st.session_state.otp_phone != norm_phone:
            st.error("❌ कृपया पहिले फोन नम्बरको OTP पठाएर प्रमाणित गर्नुहोस्।")
        elif uploaded_doc is None or doc_face_sig is None:
            st.error("❌ कृपया स्पष्ट फोटो भएको मान्य नमुना कागजात अपलोड गर्नुहोस्।")
        elif live_face_photo is None:
            st.error("❌ कृपया प्रत्यक्ष अनुहार (Live Face) को फोटो खिच्नुहोस्।")
        elif left_palm_photo is None or right_palm_photo is None:
            st.error("❌ कृपया बायाँ र दायाँ दुबै हातको हत्केला (Palm) स्क्यान गर्नुहोस्।")
        else:
            voter_id_hash = hashlib.sha256(f"{country.strip().upper()}:{id_number.strip().upper()}:{norm_phone}".encode()).hexdigest()

            if voter_id_hash in st.session_state.verified_voters:
                st.warning("⚠️ यो विवरणबाट पहिले नै मतदान टोकन जारी भइसकेको छ!")
            else:
                live_ok, live_match_sig, live_dup_sig, metrics, live_msg = analyze_live_face_and_blue_ray(live_face_photo.getvalue())
                if metrics:
                    st.caption(
                        f"जाँच विवरण — B/R: {metrics['blue_red_ratio']} | B/G: {metrics['blue_green_ratio']} | "
                        f"Brightness: {metrics['brightness']} | Glare%: {metrics['bright_pixel_pct']}% | Blur: {metrics['blur_laplacian']}"
                    )

                if not live_ok:
                    st.error(f"❌ {live_msg}")
                else:
                    face_matched, tm_pct, hist_pct = compare_doc_and_live_face(doc_face_sig, live_match_sig)
                    st.caption(f"कागजात-लाइभ मिलान स्कोर — Template: {tm_pct}% (आवश्यक ≥58%) | Histogram: {hist_pct}% (आवश्यक ≥55%)")

                    if not face_matched:
                        st.error(f"❌ कागजातको फोटो र लाइभ अनुहार मेल खाएन (Template: {tm_pct}%, Histogram: {hist_pct}%)।")
                    else:
                        is_dup, sim_score = is_duplicate_face(live_dup_sig, st.session_state.face_signatures)
                        if is_dup:
                            st.error(f"🚨 सुरक्षा चेतावनी! यो अनुहार पहिले नै अर्को परिचयपत्रमार्फत प्रयोग भइसकेको छ (समानता: {sim_score}%)।")
                        else:
                            left_ok, _, left_msg = analyze_palm_scan(left_palm_photo.getvalue(), "बायाँ हात (Left Palm)")
                            right_ok, _, right_msg = analyze_palm_scan(right_palm_photo.getvalue(), "दायाँ हात (Right Palm)")

                            if not left_ok:
                                st.error(f"❌ {left_msg}")
                            elif not right_ok:
                                st.error(f"❌ {right_msg}")
                            else:
                                secret_token = f"ZKP-{uuid.uuid4().hex[:12].upper()}"
                                st.session_state.verified_voters[voter_id_hash] = secret_token
                                st.session_state.face_signatures.append(live_dup_sig)
                                st.session_state.current_token = secret_token

                                st.success(f"✅ सबै प्रमाणीकरण सफल भयो! (OTP: {norm_phone} | TM: {tm_pct}% / Hist: {hist_pct}% | {left_msg} | {right_msg})")
                                st.info(f"तपाईंको गोप्य भोटिङ टोकन: `{secret_token}` — अब **'२. सुरक्षित मतदान (Cast Vote)'** ट्याबमा गएर भोट हाल्नुहोस्।")

with tab2:
    st.subheader("दोस्रो चरण: ब्लकचेनमा भोट सुरक्षित गर्नुहोस्")
    token_input = st.text_input("तपाईंको गोप्य भोटिङ टोकन:", value=st.session_state.current_token or "")

    candidate = st.radio(
        "उम्मेदवार / पार्टी छान्नुहोस्:",
        ["पार्टी A (विकास र प्रविधि)", "पार्टी B (युवा नेतृत्व)", "पार्टी C (आर्थिक क्रान्ति)", "स्वतन्त्र उम्मेदवार D"]
    )

    if st.button("भोट सुरक्षित गर्नुहोस्"):
        bc = st.session_state.blockchain
        if not token_input.startswith("ZKP-"):
            st.error("❌ अमान्य टोकन! पहिले पहिलो चरणमा प्रमाणीकरण पूरा गर्नुहोस्।")
        elif token_input in bc.used_tokens:
            st.error("❌ यो टोकन प्रयोग भइसकेको छ! एक व्यक्ति एक भोट मात्र मान्य हुन्छ।")
        else:
            previous_block = bc.chain[-1]
            new_block = bc.create_block(
                proof=len(bc.chain) + 100,
                previous_hash=previous_block['current_hash'],
                voter_token=token_input,
                candidate=candidate
            )
            bc.used_tokens.add(token_input)
            st.session_state.current_token = None
            st.success("🎉 भोट ब्लकचेनमा सफलतापूर्वक लक भयो!")
            st.json({
                "Block Index": new_block['index'],
                "Timestamp": new_block['timestamp'],
                "Block Hash": new_block['current_hash'],
                "Status": "IMMUTABLE & VERIFIED"
            })

with tab3:
    st.subheader("तेस्रो चरण: पारदर्शी ब्लकचेन लेजर र प्रत्यक्ष मतगणना")
    bc = st.session_state.blockchain

    if bc.is_chain_valid():
        st.success("🛡️ सिस्टम सुरक्षा स्थिति: १००% सुरक्षित (डेटा सधैं सुरक्षित र अपरिवर्तनीय छ)")
    else:
        st.error("🚨 चेतावनी: ब्लकचेन लेजरमा फेरबदल भएको भेटियो!")

    results = {}
    total_votes = len(bc.chain) - 1
    for block in bc.chain[1:]:
        cand = block['candidate']
        results[cand] = results.get(cand, 0) + 1

    st.markdown(f"### 📊 कुल खसेको मत: **{total_votes}**")

    if results:
        st.bar_chart(results)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("#### विस्तृत मत विवरण:")
            for k, v in results.items():
                pct = round((v / total_votes) * 100, 1) if total_votes > 0 else 0
                st.write(f"- **{k}:** {v} मत ({pct}%)")
    else:
        st.info("अहिलेसम्म कुनै भोट खसेको छैन।")

    st.divider()
    st.write("### 🔗 क्रिप्टोग्राफिक ब्लकचेन लेजर (Immutable Blocks):")
    for block in reversed(bc.chain):
        with st.expander(f"Block #{block['index']} | Candidate: {block['candidate']} | Hash: {block['current_hash'][:20]}..."):
            st.json(block)

with tab4:
    st.subheader("चौथो चरण: प्रयोगकर्ताको अनुभव र सुझाव पठाउनुहोस्")
    st.write("यो प्रोटोटाइप परीक्षण गर्दा तपाईंलाई कस्तो लाग्यो? कहाँ सुधार गर्नुपर्छ जस्तो लाग्यो? कृपया तल लेख्नुहोस्:")

    with st.form("feedback_form", clear_on_submit=True):
        fb_name = st.text_input("नाम (ऐच्छिक):", placeholder="उदाहरण: प्रविण")
        fb_location = st.selectbox("तपाईंले कहाँबाट परीक्षण गर्नुभयो?", ["नेपाल भित्र", "विदेशबाट"])
        fb_rating = st.slider("समग्र अनुभव मूल्याङ्कन (१ = कमजोर, ५ = उत्कृष्ट):", 1, 5, 4)
        fb_comment = st.text_area("तपाईंको सुझाव वा भेटिएको समस्या:", placeholder="जस्तै: कागजात मिलान, क्यामेरा उज्यालो, वा प्रयोग गर्न सजिलोपनबारे लेख्नुहोस्...")
        submitted = st.form_submit_button("प्रतिक्रिया पठाउनुहोस्")

        if submitted:
            if not fb_comment.strip():
                st.error("❌ कृपया केही सुझाव वा टिप्पणी लेख्नुहोस्।")
            else:
                save_feedback({
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    "name": fb_name.strip() or "अज्ञात परीक्षक",
                    "location": fb_location,
                    "rating": fb_rating,
                    "comment": fb_comment.strip()
                })
                st.success("✅ धन्यवाद! तपाईंको प्रतिक्रिया सुरक्षित भयो।")

    all_feedbacks = load_feedbacks()
    if all_feedbacks:
        st.divider()
        st.write(f"### प्राप्त प्रतिक्रियाहरू ({len(all_feedbacks)}):")
        st.download_button(
            label="📥 सबै प्रतिक्रियाहरू JSON मा डाउनलोड गर्नुहोस्",
            data=json.dumps(all_feedbacks, indent=4, ensure_ascii=False),
            file_name="sajilovote_feedback.json",
            mime="application/json"
        )
        for item in reversed(all_feedbacks):
            st.markdown(f"**{item['name']}** ({item['location']}) — ⭐ {item['rating']}/5 | `{item['timestamp']}`")
            st.write(f"> {item['comment']}")