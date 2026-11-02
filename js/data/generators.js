import { Rng, shapeSize } from '../core/tensor.js';
import { createDataset, buildSplits, recompute } from './dataset.js';
import { hashFeatures, buildCharVocab, encodeChars, buildWordVocab, CharMarkov, WordMarkov, tokenizeWords, applyTypo, wordDrop } from './text.js';
import { logMelFrames, normalizeFeatures, linearPowerFrames, mfccLite, onsetEnvelope, resample, addNoise, shiftPitch, wavEncode, peakNormalize } from './audio.js';
import {
  synthSpeechLike, synthKeyword, WAKE_PHRASES, NEGATIVE_PHRASES, soundEvent,
  rhythmTrack, synthNoise, mixSignals, synthTone
} from './synth-audio.js';
import {
  synthNatural, synthFace, synthHand, synthSubject, synthSaliency, stylize, explicitLikePattern,
  bilinearResize, bicubicResize, addGaussianNoise, jpegQuantize, clamp01, areaDownsample,
  fillPolygon, drawLine, fillRect, valueNoise, fillEllipse, gaussianBlur
} from './image.js';
import {
  IMU_ACTIVITIES, imuWindow, forecastSeries, anomalyWindow, mouseTrace, keystrokeTrace, behaviorFeatures
} from './signal.js';
import { stratifiedSplit, randomSplit } from './split.js';
import { DOMAIN_GENERATORS, DOMAIN_REFERENCES } from './domains.js';

export const CC0 = {
  id: 'generated-local',
  name: 'Generated on your machine',
  url: '',
  allowsCommercial: true,
  notes: 'Generated locally from a deterministic seed. Verify applicable rights, privacy and other obligations for your intended use.'
};

function ref(name, license, url, note) {
  return { name, license, url, note };
}

export const REFERENCE_DATASETS = {
  langid: ref('Common Voice / UDHR-based text corpora', { id: 'cc-by-4.0', name: 'CC BY 4.0', allowsCommercial: true }, 'https://commonvoice.mozilla.org', 'Attribution required. Verify per-language subsets before shipping.'),
  intent: ref('CLINC150 / SNIPS / MASSIVE', { id: 'varies', name: 'Mixed — check per corpus', allowsCommercial: false }, 'https://github.com/clinc/oos-eval', 'SNIPS is research-only. CLINC150 is CDLA-Permissive. MASSIVE is CC BY 4.0.'),
  spam: ref('Enron-Spam / SMS Spam Collection', { id: 'uci-noncommercial', name: 'UCI — research use', allowsCommercial: false }, 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection', 'Not cleared for commercial products. Toxicity labels need a human review pass.'),
  pii: ref('Synthetic PII corpora / your own logs', { id: 'own-data', name: 'Your own data', allowsCommercial: true }, '', 'Real logs need a lawful basis and redaction before they enter a training set.'),
  code: ref('The Stack v2 / per-repo sources', { id: 'varies', name: 'Per-file licenses', allowsCommercial: false }, 'https://huggingface.co/datasets/bigcode/the-stack-v2', 'Filter by license field. Many permissive files, some restricted.'),
  password: ref('Leaked password corpora', { id: 'do-not-use', name: 'Prohibited', allowsCommercial: false }, '', 'Never train on leaked credentials. Use a rule-based generator or a licensed synthetic set.'),
  embeddings: ref('Quora Question Pairs / PAWS', { id: 'varies', name: 'Mixed', allowsCommercial: false }, 'https://huggingface.co/datasets/paws', 'Quora ToS restricts commercial reuse; PAWS is CC BY-SA 4.0.'),
  lm: ref('WikiText-2 / TinyStories', { id: 'cc-by-sa', name: 'CC BY-SA 4.0', allowsCommercial: true }, 'https://huggingface.co/datasets/wikitext', 'Share-alike applies to the corpus, not to your model weights.'),
  wake: ref('Google Speech Commands', { id: 'cc-by-4.0', name: 'CC BY 4.0', allowsCommercial: true }, 'https://www.tensorflow.org/datasets/catalog/speech_commands', 'Commercial use allowed with attribution.'),
  vad: ref('LibriSpeech / your own recordings', { id: 'cc-by-4.0', name: 'CC BY 4.0', allowsCommercial: true }, 'https://www.openslr.org/12', 'Voice data may also carry privacy obligations.'),
  denoise: ref('DNS Challenge / DEMAND', { id: 'cdla-permissive', name: 'CDLA-Permissive / CC', allowsCommercial: true }, 'https://github.com/microsoft/DNS-Challenge', 'Check the specific noise subset; some are research-only.'),
  pitch: ref('MedleyDB / MDB-stem-synth', { id: 'cc-by-nc', name: 'CC BY-NC 4.0', allowsCommercial: false }, 'https://medleydb.weebly.com', 'Non-commercial. Record your own stems for a shippable tuner.'),
  events: ref('AudioSet / UrbanSound8K / FSD50K', { id: 'mixed', name: 'Mixed', allowsCommercial: false }, 'https://research.google.com/audioset/', 'AudioSet labels are CC BY 4.0 but source clips are not. UrbanSound8K is CC BY-NC. FSD50K is CC BY 4.0.'),
  onset: ref('GTZAN / Ballroom / your own audio', { id: 'mixed', name: 'Mixed', allowsCommercial: false }, 'https://huggingface.co/datasets/marsyas/gtzan', 'GTZAN is research-only. Ballroom is CC BY-NC.'),
  superres: ref('DIV2K / Flickr2K / Open Images', { id: 'mixed', name: 'DIV2K research, Open Images CC BY 4.0', allowsCommercial: true }, 'https://data.vision.ee.ethz.ch/cvl/DIV2K/', 'Open Images is cleared for commercial use with attribution; DIV2K is not.'),
  denoiseimg: ref('SIDD / PolyU / Open Images', { id: 'mixed', name: 'Mixed', allowsCommercial: false }, 'https://www.eecs.yorku.ca/~kamel/sidd/', 'SIDD is research-only. Generate your own pairs for shipping.'),
  detection: ref('WIDER FACE / HaGRID / EgoHands', { id: 'mixed', name: 'Mixed', allowsCommercial: false }, 'https://shuoyang1213.me/WIDERFACE/', 'WIDER FACE is non-commercial research only. HaGRID is CC BY 4.0.'),
  segmentation: ref('MediaPipe Selfie / Ego4D', { id: 'apache-2.0', name: 'Apache 2.0 (dataset terms vary)', allowsCommercial: true }, 'https://developers.google.com/mediapipe', 'Model code is Apache 2.0; verify the training set terms separately.'),
  style: ref('WikiArt / open-licensed art', { id: 'mixed', name: 'Public domain + CC', allowsCommercial: true }, 'https://www.wikiart.org', 'Use only public-domain or CC-BY works for a commercial style model.'),
  saliency: ref('SALICON / MSRA10K', { id: 'mixed', name: 'Mixed', allowsCommercial: false }, 'http://salicon.net', 'SALICON is CC BY-NC-SA. MSRA10K is research-only.'),
  nsfw: ref('NudeNet dataset / your own moderated corpus', { id: 'mixed', name: 'Restricted', allowsCommercial: false }, 'https://github.com/notAI-tech/NudeNet', 'Adult-content classifiers carry platform and jurisdictional obligations. Human review is mandatory.'),
  imu: ref('UCI HAR / WISDM / MotionSense', { id: 'mixed', name: 'UCI research / CC BY 4.0', allowsCommercial: true }, 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones', 'MotionSense is CC BY 4.0; UCI HAR terms are looser.'),
  anomaly: ref('Numenta / SMD / your own telemetry', { id: 'apache-2.0', name: 'Apache 2.0 / own data', allowsCommercial: true }, 'https://github.com/numenta/NAB', 'Your own production telemetry is usually the best and safest source.'),
  forecast: ref('M4 / M5 / ETTh', { id: 'mixed', name: 'CC BY 4.0 / research', allowsCommercial: true }, 'https://github.com/Mcompetitions/M4-methods', 'M4 is CC BY 4.0. ETTh is research-only.'),
  bot: ref('Your own session telemetry', { id: 'own-data', name: 'Your own data', allowsCommercial: true }, '', 'Behavioural biometrics are personal data under GDPR — document the lawful basis.'),
  compression: ref('enwik8 / Calgary Corpus / your own assets', { id: 'public-domain', name: 'Public domain / own data', allowsCommercial: true }, 'https://mattmahoney.net/dc/textdata.html', 'enwik8 is a Wikipedia extract under CC BY-SA; check before redistributing.'),
  texture: ref('Your own image assets', { id: 'own-data', name: 'Your own data', allowsCommercial: true }, '', 'A neural texture encodes the image itself — license it like the source asset.')
};

const LANG_CORPUS = {
  english: ['the quick brown fox jumps over the lazy dog', 'we should meet tomorrow morning to review the numbers', 'she does not know where the keys were left', 'there is a problem with the account settings', 'they have been working on this project for months', 'please send the report before the end of the day', 'i would like to order something for delivery', 'the weather was much better last week', 'he asked whether anyone had seen the file', 'this is the best result we have had so far', 'the server responded slowly because the cache had been cleared overnight', 'can you confirm whether the invoice was paid on time', 'we need a decision from the board before friday afternoon', 'the new build failed twice but the third attempt succeeded', 'nobody remembered to lock the door when they left', 'she explained how the interest rate affects the monthly payment', 'there were too many people in the meeting room to sit down', 'i will call you back as soon as i have the answer', 'the train was delayed by forty minutes due to signals failure', 'please keep the receipt in case you want a refund later', 'his brother moved abroad and found work within a week', 'we should write down the assumptions before running the model'],
  spanish: ['el rapido zorro marron salta sobre el perro perezoso', 'debemos reunirnos manana por la manana para revisar las cifras', 'ella no sabe donde quedaron las llaves', 'hay un problema con la configuracion de la cuenta', 'han estado trabajando en este proyecto durante meses', 'por favor envia el informe antes de que termine el dia', 'me gustaria pedir algo para entregar en casa', 'el tiempo fue mucho mejor la semana pasada', 'el pregunto si alguien habia visto el archivo', 'este es el mejor resultado que hemos tenido hasta ahora', 'el servidor respondio lentamente porque la memoria se borro durante la noche', 'puedes confirmar si la factura fue pagada a tiempo', 'necesitamos una decision de la junta antes del viernes por la tarde', 'nadie recordo cerrar la puerta con llave cuando se fueron', 'ella explico como la tasa de interes afecta el pago mensual', 'habia demasiada gente en la sala de reuniones para sentarse', 'te llamare en cuanto tenga la respuesta', 'el tren se retraso cuarenta minutos por un fallo de senales', 'guarda el recibo por si quieres un reembolso mas tarde', 'su hermano se mudo al extranjero y encontro trabajo en una semana', 'debemos escribir las suposiciones antes de ejecutar el modelo', 'la reunion comenzo tarde porque faltaban dos personas'],
  french: ['le rapide renard brun saute par dessus le chien paresseux', 'nous devrions nous retrouver demain matin pour examiner les chiffres', 'elle ne sait pas ou les cles ont ete laissees', 'il y a un probleme avec les parametres du compte', 'ils travaillent sur ce projet depuis plusieurs mois', 'veuillez envoyer le rapport avant la fin de la journee', 'je voudrais commander quelque chose pour la livraison', 'la meteo etait bien meilleure la semaine derniere', 'il a demande si quelqu un avait vu le dossier', 'c est le meilleur resultat que nous ayons eu jusqu ici', 'le serveur a repondu lentement parce que le cache avait ete vide pendant la nuit', 'pouvez vous confirmer si la facture a ete payee a temps', 'il nous faut une decision du conseil avant vendredi apres midi', 'personne n a pense a fermer la porte a cle en partant', 'elle a explique comment le taux d interet influence la mensualite', 'il y avait trop de monde dans la salle pour s asseoir', 'je vous rappelle des que j ai la reponse', 'le train a eu quarante minutes de retard a cause des signaux', 'gardez le recu au cas ou vous voudriez un remboursement', 'son frere est parti a l etranger et a trouve du travail en une semaine', 'nous devrions noter les hypotheses avant de lancer le modele', 'la reunion a commence en retard car il manquait deux personnes'],
  german: ['der schnelle braune fuchs springt ueber den faulen hund', 'wir sollten uns morgen frueh treffen um die zahlen zu pruefen', 'sie weiss nicht wo die schluessel geblieben sind', 'es gibt ein problem mit den kontoeinstellungen', 'sie arbeiten seit monaten an diesem projekt', 'bitte senden sie den bericht vor ende des tages', 'ich moechte etwas zur lieferung bestellen', 'das wetter war letzte woche viel besser', 'er fragte ob jemand die datei gesehen hat', 'das ist das beste ergebnis das wir bisher hatten', 'der server antwortete langsam weil der cache ueber nacht geleert wurde', 'koennen sie bestaetigen ob die rechnung rechtzeitig bezahlt wurde', 'wir brauchen eine entscheidung des vorstands bis freitag nachmittag', 'niemand dachte daran die tuer abzuschliessen als sie gingen', 'sie erklaerte wie der zinssatz die monatliche rate beeinflusst', 'es waren zu viele leute im besprechungsraum zum sitzen', 'ich rufe zurueck sobald ich die antwort habe', 'der zug hatte vierzig minuten verspaetung wegen einer signalstoerung', 'bitte bewahren sie den beleg fuer eine moegliche rueckerstattung auf', 'sein bruder zog ins ausland und fand innerhalb einer woche arbeit', 'wir sollten die annahmen notieren bevor wir das modell starten', 'die besprechung begann spaet weil zwei personen fehlten'],
  portuguese: ['a rapida raposa marrom salta sobre o cao preguicoso', 'devemos nos reunir amanha de manha para rever os numeros', 'ela nao sabe onde as chaves foram deixadas', 'existe um problema com as configuracoes da conta', 'eles tem trabalhado neste projeto ha varios meses', 'por favor envie o relatorio antes do fim do dia', 'eu gostaria de pedir algo para entrega', 'o tempo estava muito melhor na semana passada', 'ele perguntou se alguem tinha visto o arquivo', 'este e o melhor resultado que tivemos ate agora', 'o servidor respondeu devagar porque a memoria foi limpa durante a noite', 'voce pode confirmar se a fatura foi paga no prazo', 'precisamos de uma decisao da diretoria antes de sexta a tarde', 'ninguem lembrou de trancar a porta quando sairam', 'ela explicou como a taxa de juros afeta a parcela mensal', 'havia gente demais na sala de reunioes para sentar', 'eu ligo de volta assim que tiver a resposta', 'o trem atrasou quarenta minutos por falha de sinalizacao', 'guarde o recibo caso queira um reembolso depois', 'o irmao dele mudou para o exterior e encontrou trabalho em uma semana', 'devemos anotar as premissas antes de executar o modelo', 'a reuniao comecou atrasada porque faltavam duas pessoas'],
  italian: ['la veloce volpe marrone salta sopra il cane pigro', 'dovremmo incontrarci domani mattina per rivedere i numeri', 'lei non sa dove sono state lasciate le chiavi', 'c e un problema con le impostazioni dell account', 'stanno lavorando a questo progetto da mesi', 'per favore invia il rapporto prima della fine della giornata', 'vorrei ordinare qualcosa per la consegna', 'il tempo era molto meglio la settimana scorsa', 'ha chiesto se qualcuno aveva visto il file', 'questo e il miglior risultato che abbiamo avuto finora', 'il server ha risposto lentamente perche la cache era stata svuotata di notte', 'puoi confermare se la fattura e stata pagata in tempo', 'ci serve una decisione del consiglio prima di venerdi pomeriggio', 'nessuno ha ricordato di chiudere la porta a chiave quando sono usciti', 'lei ha spiegato come il tasso di interesse influisce sulla rata mensile', 'c erano troppe persone nella sala riunioni per sedersi', 'ti richiamo appena ho la risposta', 'il treno ha accumulato quaranta minuti di ritardo per un guasto ai segnali', 'conserva lo scontrino nel caso volessi un rimborso piu tardi', 'suo fratello si e trasferito all estero e ha trovato lavoro in una settimana', 'dovremmo annotare le ipotesi prima di eseguire il modello', 'la riunione e iniziata in ritardo perche mancavano due persone'],
  dutch: ['de snelle bruine vos springt over de luie hond', 'we moeten elkaar morgenochtend ontmoeten om de cijfers te bekijken', 'zij weet niet waar de sleutels zijn gebleven', 'er is een probleem met de accountinstellingen', 'ze werken al maanden aan dit project', 'stuur het rapport voor het einde van de dag', 'ik wil graag iets bestellen om te laten bezorgen', 'het weer was vorige week veel beter', 'hij vroeg of iemand het bestand had gezien', 'dit is het beste resultaat dat we tot nu toe hadden', 'de server reageerde traag omdat de cache s nachts was geleegd', 'kun je bevestigen of de factuur op tijd is betaald', 'we hebben vrijdagmiddag een besluit van het bestuur nodig', 'niemand dacht eraan de deur op slot te doen toen ze vertrokken', 'ze legde uit hoe de rente de maandelijkse betaling beinvloedt', 'er waren te veel mensen in de vergaderzaal om te zitten', 'ik bel je terug zodra ik het antwoord heb', 'de trein had veertig minuten vertraging door een sein storing', 'bewaar de bon voor het geval je geld terug wilt', 'zijn broer verhuisde naar het buitenland en vond binnen een week werk', 'we moeten de aannames opschrijven voordat we het model draaien', 'de vergadering begon laat omdat er twee mensen ontbraken'],
  polish: ['szybki brazowy lis przeskakuje nad leniwym psem', 'powinnismy spotkac sie jutro rano aby przejrzec liczby', 'ona nie wie gdzie zostaly zostawione klucze', 'jest problem z ustawieniami konta', 'pracuja nad tym projektem od kilku miesiecy', 'prosz wyslac raport przed koncem dnia', 'chcialbym zamowic cos z dostawa', 'pogoda byla duzo lepsza w zeszlym tygodniu', 'pytal czy ktos widzial ten plik', 'to najlepszy wynik jaki do tej pory uzyskalismy', 'serwer odpowiadal wolno poniewaz pamiec podreczna zostala wyczyszczona w nocy', 'czy mozesz potwierdzic czy faktura zostala zaplacona na czas', 'potrzebujemy decyzji zarzadu przed piatkiem po poludniu', 'nikt nie pamietal o zamknieciu drzwi kiedy wychodzili', 'wyjasnila jak stopa procentowa wplywa na miesieczna rate', 'w sali konferencyjnej bylo zbyt duzo ludzi zeby usiasc', 'oddzwonie jak tylko bede miec odpowiedz', 'pociag mial czterdziesci minut opoznienia z powodu awarii sygnalizacji', 'zachowaj paragon na wypadek zwrotu pieniedzy', 'jego brat wyjechal za granice i znalazl prace w ciagu tygodnia', 'powinnismy zapisac zalozenia przed uruchomieniem modelu', 'spotkanie zaczelo sie pozno poniewaz brakowalo dwoch osob'],
  turkish: ['hizli kahverengi tilki tembel kopegin uzerinden atlar', 'rakamlari gozden gecirmek icin yarin sabah bulusmaliyiz', 'anahtarlarin nerede oldugunu bilmiyor', 'hesap ayarlarinda bir sorun var', 'aylardir bu proje uzerinde calisiyorlar', 'lutfen raporu gun bitmeden gonderin', 'teslimat icin bir sey siparis etmek istiyorum', 'hava gecen hafta cok daha iyiydi', 'dosyayi birinin gorup gormedigini sordu', 'su ana kadar aldigimiz en iyi sonuc bu', 'sunucu yavas yanit verdi cunku onbellek gece boyunca temizlenmisti', 'faturanin zamaninda odenip odenmedigini onaylayabilir misin', 'cuma ogleden once yonetim kurulundan bir karar gerekiyor', 'ciktiklarinda kimse kapiyi kilitlemayi hatirlamadi', 'faiz oraninin aylik odemeyi nasil etkiledigini acikladi', 'toplanti salonunda oturmak icin cok fazla insan vardi', 'cevabi alir almaz seni geri arayacagim', 'tren sinyal arizasi yuzunden kirk dakika gecikti', 'iade istersen diye fisi sakla', 'kardesi yurt disina tasindi ve bir hafta icinde is buldu', 'modeli calistirmadan once varsayimlari yazmaliyiz', 'toplanti iki kisi eksik oldugu icin gec basladi'],
  indonesian: ['rubah cokelat yang cepat melompati anjing malas', 'kita harus bertemu besok pagi untuk meninjau angka angka', 'dia tidak tahu di mana kunci itu ditinggalkan', 'ada masalah dengan pengaturan akun', 'mereka telah mengerjakan proyek ini selama berbulan bulan', 'silakan kirim laporan sebelum akhir hari', 'saya ingin memesan sesuatu untuk diantar', 'cuaca jauh lebih baik minggu lalu', 'dia bertanya apakah ada yang melihat berkas itu', 'ini hasil terbaik yang pernah kami dapatkan', 'server merespons lambat karena tembolok dibersihkan semalaman', 'bisakah kamu memastikan apakah tagihan sudah dibayar tepat waktu', 'kami perlu keputusan dewan sebelum jumat siang', 'tidak ada yang ingat mengunci pintu ketika mereka pergi', 'dia menjelaskan bagaimana suku bunga memengaruhi cicilan bulanan', 'terlalu banyak orang di ruang rapat sehingga tidak bisa duduk', 'saya akan menelepon kembali begitu mendapat jawaban', 'kereta terlambat empat puluh menit karena gangguan sinyal', 'simpan struknya kalau nanti ingin pengembalian dana', 'kakaknya pindah ke luar negeri dan mendapat pekerjaan dalam seminggu', 'kita harus mencatat asumsi sebelum menjalankan model', 'rapat dimulai terlambat karena dua orang tidak hadir'],
  vietnamese: ['con cao nau nhanh nhay qua con cho luoi', 'chung ta nen gap nhau vao sang mai de xem lai cac con so', 'co ay khong biet chia khoa duoc de o dau', 'co van de voi cai dat tai khoan', 'ho da lam viec trong du an nay nhieu thang', 'vui long gui bao cao truoc cuoi ngay', 'toi muon dat mot thu gi do de giao hang', 'thoi tiet tot hon nhieu vao tuan truoc', 'anh ay hoi co ai thay tep tin khong', 'day la ket qua tot nhat ma chung toi co duoc', 'may chu phan hoi cham vi bo nho dem da bi xoa qua dem', 'ban co the xac nhan hoa don da duoc thanh toan dung han khong', 'chung ta can quyet dinh cua hoi dong truoc chieu thu sau', 'khong ai nho khoa cua khi ho roi di', 'co ay giai thich lai suat anh huong den khoan tra hang thang nhu the nao', 'co qua nhieu nguoi trong phong hop de ngoi', 'toi se goi lai ngay khi co cau tra loi', 'chuyen tau bi tre bon muoi phut do hong tin hieu', 'hay giu lai bien lai phong khi ban muon hoan tien', 'anh trai cua anh ay ra nuoc ngoai va tim duoc viec trong mot tuan', 'chung ta nen ghi lai cac gia dinh truoc khi chay mo hinh', 'cuoc hop bat dau muon vi thieu hai nguoi'],
  swahili: ['mbweha haraka wa rangi ya kahawia anaruka juu ya mbwa mvivu', 'tunapaswa kukutana kesho asubuhi kupitia nambari', 'hajui funguo ziliachwa wapi', 'kuna tatizo na mipangilio ya akaunti', 'wamekuwa wakifanya kazi kwenye mradi huu kwa miezi', 'tafadhali tuma ripoti kabla ya mwisho wa siku', 'ningependa kuagiza kitu kwa ajili ya usafirishaji', 'hali ya hewa ilikuwa bora zaidi wiki iliyopita', 'aliuliza ikiwa mtu ameona faili', 'hii ndiyo matokeo bora tuliyopata hadi sasa', 'seva ilijibu polepole kwa sababu akiba iliwekwa wazi usiku kucha', 'unaweza kuthibitisha kama ankara ililipwa kwa wakati', 'tunahitaji uamuzi wa bodi kabla ya ijumaa jioni', 'hakuna aliyekumbuka kufunga mlango walipoondoka', 'alieleza jinsi kiwango cha riba kinavyoathiri malipo ya kila mwezi', 'kulikuwa na watu wengi sana chumbani mwaka wa mikutano kuketi', 'nitakupigia simu mara tu nitakapopata jibu', 'treni ilichelewa dakika arobaini kutokana na hitilafu ya ishara', 'weka stakabadhi endapo utataka kurejeshewa pesa baadaye', 'ndugu yake alihamia nje ya nchi na akapata kazi ndani ya wiki moja', 'tunapaswa kuandika mawazo kabla ya kuendesha mfano', 'mkutano ulianza kwa kuchelewa kwa sababu watu wawili hawakufika']
};

function sampleSentence(rng, sentences, minWords, maxWords) {
  const s = sentences[rng.int(sentences.length)];
  const words = s.split(' ');
  const n = Math.min(words.length, minWords + rng.int(Math.max(1, maxWords - minWords)));
  const start = rng.int(Math.max(1, words.length - n + 1));
  return words.slice(start, start + n).join(' ');
}

export function genLangId(opts) {
  const cfg = Object.assign({ seed: 11, size: 'medium', dim: 640, charN: [2, 5], languages: null }, opts || {});
  const rng = new Rng(cfg.seed);
  const langs = cfg.languages || Object.keys(LANG_CORPUS);
  const perClass = { small: 40, medium: 110, large: 320 }[cfg.size] || 110;
  const charModels = {};
  const wordModels = {};
  for (const l of langs) {
    charModels[l] = new CharMarkov(2, 0.5).fit(LANG_CORPUS[l]);
    wordModels[l] = new WordMarkov(2, 0.6).fit(LANG_CORPUS[l]);
  }
  const spec = { dim: cfg.dim, charN: cfg.charN, wordN: [1, 1], signed: true, norm: 'l2', lowercase: true };
  const texts = [];
  const labelsIdx = [];
  const meta = [];
  for (let li = 0; li < langs.length; li++) {
    const l = langs[li];
    const made = collectUnique(perClass, () => {
      let text;
      const roll = rng.next();
      if (roll < 0.44) {
        text = sampleSentence(rng, LANG_CORPUS[l], 4, 13);
      } else if (roll < 0.88) {
        text = wordModels[l].sample(rng, 5, 6 + rng.int(9));
      } else {
        text = charModels[l].sample(rng, 18 + rng.int(30));
      }
      if (rng.next() < 0.14) text = text.slice(0, Math.max(12, Math.floor(text.length * rng.range(0.4, 0.95))));
      if (rng.next() < 0.1) text = text.slice(Math.floor(text.length * rng.range(0.05, 0.25)));
      if (rng.next() < 0.08) text = applyTypo(text, rng, 0.04);
      if (rng.next() < 0.1) text = wordDrop(text, rng, 0.12);
      text = text.toLowerCase().replace(/\s+/g, ' ').trim();
      if (text.length < 14) return null;
      return text;
    }, (t) => t);
    for (const text of made) {
      texts.push(text);
      labelsIdx.push(li);
      meta.push({ text, preview: text.slice(0, 90), language: l });
    }
  }
  const order = rng.shuffle(texts.map((_, i) => i));
  const x = new Float32Array(texts.length * cfg.dim);
  const y = new Float32Array(texts.length);
  const orderedMeta = [];
  for (let i = 0; i < order.length; i++) {
    const src = order[i];
    x.set(hashFeatures(texts[src], spec), i * cfg.dim);
    y[i] = labelsIdx[src];
    orderedMeta.push(meta[src]);
  }
  const ds = createDataset({
    name: 'Language identification',
    slug: 'langid',
    description: 'Short text snippets across ' + langs.length + ' languages, sampled from per-language character models fit on seed sentences. Features are hashed character n-grams.',
    modality: 'text',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'langid',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    labels: langs,
    meta: orderedMeta,
    featureSpec: spec,
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'langid', languages: langs, perClass, seed: cfg.seed, reference: REFERENCE_DATASETS.langid },
    config: cfg
  });
  return ds;
}

const INTENT_TEMPLATES = {
  lights_on: ['turn on the {room} lights', 'switch the {room} light on', 'can you light up the {room}', 'i need the lights in the {room}', 'brighten the {room}'],
  lights_off: ['turn off the {room} lights', 'kill the lights in the {room}', 'switch the {room} light off', 'darken the {room} please', 'lights out in the {room}'],
  play_music: ['play {artist}', 'put on some {genre}', 'start playing {playlist}', 'i want to listen to {genre}', 'play the {playlist} playlist'],
  pause_music: ['pause the music', 'stop playing', 'hold the track', 'pause it for a second', 'stop the song'],
  volume_up: ['turn it up', 'increase the volume', 'louder please', 'raise the volume a bit', 'make it louder'],
  volume_down: ['turn it down', 'lower the volume', 'quieter please', 'reduce the volume', 'make it quieter'],
  set_timer: ['set a timer for {minutes} minutes', 'remind me in {minutes} minutes', 'start a {minutes} minute timer', 'timer {minutes} minutes', 'wake me in {minutes} minutes'],
  weather: ['what is the weather in {city}', 'will it rain in {city} today', 'how hot is it in {city}', 'weather forecast for {city}', 'do i need a coat in {city}'],
  navigate: ['take me to {place}', 'directions to {place}', 'navigate to {place}', 'how do i get to {place}', 'fastest route to {place}'],
  order_refund: ['i want a refund for order {orderid}', 'refund my order {orderid}', 'how do i get my money back for {orderid}', 'cancel and refund order {orderid}', 'the item in order {orderid} was broken i want a refund'],
  cancel_order: ['cancel order {orderid}', 'i need to cancel my order', 'stop the delivery for {orderid}', 'do not ship order {orderid}', 'cancel that purchase'],
  check_balance: ['what is my balance', 'how much money do i have', 'show my account balance', 'check the balance on my {account} account', 'tell me my remaining balance'],
  book_flight: ['book a flight to {city} on {date}', 'find me flights to {city}', 'i need a plane ticket to {city}', 'cheapest flight to {city} on {date}', 'reserve a seat to {city}'],
  thermostat_up: ['make it warmer', 'set the thermostat to {temp} degrees', 'turn the heating up', 'increase the temperature to {temp}', 'i am cold turn up the heat'],
  thermostat_down: ['make it cooler', 'set the thermostat to {temp} degrees', 'turn the air conditioning up', 'lower the temperature to {temp}', 'i am hot turn on the cooling'],
  call_contact: ['call {contact}', 'phone {contact} now', 'make a call to {contact}', 'ring {contact}', 'dial {contact}'],
  send_message: ['message {contact} that {message}', 'send a text to {contact}', 'tell {contact} {message}', 'text {contact} saying {message}', 'write to {contact}'],
  find_restaurant: ['find a {cuisine} restaurant near me', 'where can i get {cuisine} food', 'recommend a {cuisine} place', 'best {cuisine} restaurant around here', 'search for {cuisine} nearby']
};

const SLOT_VALUES = {
  room: ['kitchen', 'living room', 'bedroom', 'bathroom', 'office', 'garage', 'hallway'],
  artist: ['the rolling stones', 'miles davis', 'daft punk', 'tame impala', 'nina simone'],
  genre: ['jazz', 'techno', 'classical', 'hip hop', 'ambient', 'country'],
  playlist: ['focus', 'workout', 'dinner', 'road trip', 'sleep'],
  minutes: ['3', '5', '10', '15', '20', '25', '30', '45', '60'],
  city: ['auckland', 'lisbon', 'toronto', 'berlin', 'osaka', 'nairobi', 'dublin'],
  place: ['the airport', 'work', 'the nearest petrol station', 'the hospital', 'home'],
  orderid: ['#4417', '#8802', '#1290', '#7734', '#5561'],
  account: ['checking', 'savings', 'business'],
  date: ['monday', 'the twelfth', 'next friday', 'the first of march'],
  temp: ['18', '19', '20', '21', '22', '23', '24'],
  contact: ['mum', 'sam', 'the office', 'dr patel', 'alex'],
  message: ['i will be late', 'running five minutes behind', 'thanks for the help', 'can we reschedule'],
  cuisine: ['italian', 'thai', 'mexican', 'japanese', 'indian', 'ethiopian']
};

function fillTemplate(rng, tpl) {
  return tpl.replace(/\{(\w+)\}/g, (m, key) => {
    const vals = SLOT_VALUES[key];
    return vals ? vals[rng.int(vals.length)] : key;
  });
}

export function genIntent(opts) {
  const cfg = Object.assign({ seed: 22, size: 'medium', dim: 900, noise: 0.08 }, opts || {});
  const rng = new Rng(cfg.seed);
  const intents = Object.keys(INTENT_TEMPLATES);
  const perClass = { small: 26, medium: 62, large: 150 }[cfg.size] || 62;
  const spec = { dim: cfg.dim, charN: [2, 4], wordN: [1, 2], signed: true, norm: 'l2', lowercase: true };
  const rows = [];
  for (const intent of intents) {
    const made = collectUnique(perClass, () => {
      let text = fillTemplate(rng, INTENT_TEMPLATES[intent][rng.int(INTENT_TEMPLATES[intent].length)]);
      if (rng.next() < 0.35) text = politeness(rng) + ' ' + text;
      if (rng.next() < 0.28) text = text + ' ' + tail(rng);
      if (rng.next() < 0.18) text = applyTypo(text, rng, 0.06);
      if (rng.next() < cfg.noise) text = jitterWords(text, rng);
      if (rng.next() < 0.12) text = text.replace(/\bi will\b/g, "i'll").replace(/\bdo not\b/g, "don't").replace(/\bcan not\b/g, "can't");
      return { text: text.trim().toLowerCase(), intent };
    }, (r) => r.text);
    for (const r of made) rows.push(r);
  }
  const shuffled = rng.shuffle(rows);
  const x = new Float32Array(shuffled.length * cfg.dim);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(hashFeatures(r.text, spec), i * cfg.dim);
    y[i] = intents.indexOf(r.intent);
    meta.push({ text: r.text, preview: r.text, intent: r.intent });
  });
  return createDataset({
    name: 'Intent classifier utterances',
    slug: 'intent',
    description: intents.length + ' command intents generated from slot-filled templates with optional politeness prefixes, tails and word-order noise.',
    modality: 'text',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'intent',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    labels: intents,
    meta,
    featureSpec: spec,
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'intent', intents, perClass, seed: cfg.seed, reference: REFERENCE_DATASETS.intent },
    config: cfg
  });
}

function politeness(rng) {
  return rng.pick(['please', 'hey', 'ok so', 'could you', 'i want to', 'can you', 'right', 'um', 'hello']);
}

function tail(rng) {
  return rng.pick(['thanks', 'please', 'right now', 'asap', 'if you can', 'cheers', 'now']);
}

function jitterWords(text, rng) {
  const words = text.split(' ');
  if (words.length < 3) return text;
  const i = rng.int(words.length - 1);
  const t = words[i];
  words[i] = words[i + 1];
  words[i + 1] = t;
  return words.join(' ');
}

const HAM_TEMPLATES = [
  'hey are we still on for lunch on thursday',
  'the deployment finished at 3am and all checks passed',
  'can you send me the agenda before the meeting',
  'i left my jacket in the car i will grab it later',
  'the report is attached let me know what you think',
  'mom says dinner is at seven do not be late',
  'we should book the flights before prices go up',
  'the test suite is failing on the login flow again',
  'thanks for helping me move on saturday',
  'the invoice for march has been paid in full',
  'did you see the email about the new office hours',
  'i am running about ten minutes behind sorry',
  'the garden needs watering before it rains',
  'your order has shipped and will arrive on tuesday',
  'please review the pull request when you get a chance'
];

const SPAM_TEMPLATES = [
  'you have been selected for a cash prize claim now',
  'urgent your account will be suspended verify immediately',
  'make 5000 a week from home no experience needed',
  'congratulations winner click the link to collect',
  'cheap pharmacy no prescription required order today',
  'your package could not be delivered pay the fee here',
  'limited time offer lose 20kg in 30 days guaranteed',
  'final notice regarding your unpaid invoice act now',
  'hot singles in your area are waiting to meet you',
  'double your investment in 24 hours risk free',
  'your warranty has expired renew for only 99',
  'work from home opportunity no interview required apply now',
  'you have 1 unclaimed reward confirm your details',
  'bank alert unusual activity confirm your password now'
];

const TOXIC_TEMPLATES = [
  'you are completely useless and everyone knows it',
  'shut up nobody asked for your stupid opinion',
  'what an absolute waste of space you are',
  'go away you pathetic little troll',
  'i hope your project fails you incompetent fool',
  'this is the dumbest thing i have read all year',
  'you clearly have no idea what you are talking about idiot',
  'delete your account nobody will miss you',
  'disgusting behaviour you should be ashamed',
  'stop posting this garbage you absolute clown',
  'nobody cares about you or your dumb complaints',
  'you are a disgrace and everyone is laughing at you'
];

export function genSpamToxicity(opts) {
  const cfg = Object.assign({ seed: 33, size: 'medium', dim: 768, classes: 3, labelNoise: 0.02 }, opts || {});
  const rng = new Rng(cfg.seed);
  const perClass = { small: 60, medium: 150, large: 380 }[cfg.size] || 150;
  const spec = { dim: cfg.dim, charN: [2, 5], wordN: [1, 2], signed: true, norm: 'l2', lowercase: true };
  const rows = [];
  const sources = [
    { label: 0, name: 'ham', list: HAM_TEMPLATES },
    { label: 1, name: 'spam', list: SPAM_TEMPLATES },
    { label: 2, name: 'toxic', list: TOXIC_TEMPLATES }
  ];
  for (const src of sources) {
    if (cfg.classes === 2 && src.label === 2) continue;
    const made = collectUnique(perClass, () => {
      let text = src.list[rng.int(src.list.length)];
      if (rng.next() < 0.34) text = rng.pick(['hey ', 'ok ', 'so ', 'wow ', 're: ', 'fw: ', 'urgent ', '']) + text;
      if (rng.next() < 0.3) text = text + rng.pick([' !!!', '...', ' now', ' today', ' please', ' asap', ' 100%', ' - click here', ' limited time']);
      if (rng.next() < 0.22) text = text + ' ' + rng.pick(['reply stop to opt out', 'no obligation', 'act fast', 'see you there', 'thanks', 'let me know', 'sent from my phone']);
      if (rng.next() < 0.16) text = text.split(' ').map((w) => (rng.next() < 0.25 ? w.toUpperCase() : w)).join(' ');
      if (rng.next() < 0.12) text = text.replace(/\b(\d{2,4})\b/g, (m) => String(parseInt(m, 10) + rng.int(900) + 10));
      if (rng.next() < 0.1) text = applyTypo(text, rng, 0.05);
      const flipped = rng.next() < cfg.labelNoise;
      return { text: text.trim(), label: cfg.classes === 2 ? Math.min(src.label, 1) : src.label, name: src.name, flipped };
    }, (r) => r.text);
    for (const r of made) rows.push(r);
  }
  const shuffled = rng.shuffle(rows);
  const x = new Float32Array(shuffled.length * cfg.dim);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(hashFeatures(r.text, spec), i * cfg.dim);
    y[i] = r.label;
    meta.push({ text: r.text, preview: r.text.slice(0, 90), source: r.name });
  });
  return createDataset({
    name: cfg.classes === 2 ? 'Spam filter messages' : 'Spam and toxicity messages',
    slug: 'spam-toxicity',
    description: cfg.classes === 2
      ? 'Ham versus spam messages for a client-side filter. Composed from greeting, body and sign-off fragments so no two messages are identical.'
      : 'Three-way ham / spam / toxic messages. Deliberate label noise of ' + Math.round(cfg.labelNoise * 100) + '% is injected so the achievable ceiling is visible in the confusion matrix.',
    modality: 'text',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'spam',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    labels: cfg.classes === 2 ? ['ham', 'spam'] : ['ham', 'spam', 'toxic'],
    meta,
    featureSpec: spec,
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'spam', perClass, labelNoise: cfg.labelNoise, seed: cfg.seed, reference: REFERENCE_DATASETS.spam },
    config: cfg
  });
}

const PII_NAMES = ['sarah connor', 'james okoye', 'maria silva', 'li wei', 'anna kowalski', 'tom bradley', 'yuki tanaka', 'omar haddad'];
const PII_STREETS = ['12 baker street', '88 queen road', '45 elm avenue', '7 harbour lane', '301 king parade'];
const PII_CITIES = ['auckland', 'leeds', 'porto', 'osaka', 'nairobi'];

export function genPii(opts) {
  const cfg = Object.assign({ seed: 44, size: 'medium', maxLen: 96, piiRate: 0.62 }, opts || {});
  const rng = new Rng(cfg.seed);
  const perClass = { small: 150, medium: 430, large: 1050 }[cfg.size] || 430;
  const tags = ['O', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE', 'B-CARD', 'I-CARD', 'B-KEY', 'I-KEY', 'B-NAME', 'I-NAME', 'B-ADDRESS', 'I-ADDRESS', 'B-DATE', 'I-DATE'];
  const tagIndex = {};
  tags.forEach((t, i) => { tagIndex[t] = i; });
  const fillers = [
    'please forward this to the team', 'the customer said', 'we need to check', 'according to the logs',
    'i have attached the details', 'can you confirm', 'the account belongs to', 'send the receipt to',
    'the api call failed for', 'contact support at', 'the delivery address is', 'the payment card used was',
    'this token is used by the service', 'the booking was made on'
  ];
  const texts = [];
  const tagSeqs = [];
  for (let i = 0; i < perClass; i++) {
    const parts = [];
    const tseq = [];
    const push = (text, tag) => {
      const words = text.split(' ');
      words.forEach((w, wi) => {
        parts.push(w);
        if (tag === 'O') tseq.push('O');
        else tseq.push(wi === 0 ? 'B-' + tag : 'I-' + tag);
      });
    };
    push(fillers[rng.int(fillers.length)], 'O');
    const kinds = ['EMAIL', 'PHONE', 'NAME', 'CARD', 'KEY', 'ADDRESS', 'DATE'];
    const count = 1 + (rng.next() < cfg.piiRate ? rng.int(2) : 0);
    const chosen = rng.shuffle(kinds).slice(0, count);
    for (const kind of chosen) {
      if (rng.next() < 0.3) push(rng.pick(['and', 'also', 'then', 'so']), 'O');
      switch (kind) {
        case 'EMAIL': push(rng.pick(PII_NAMES).replace(' ', '.') + rng.int(99) + '@' + rng.pick(['example.com', 'mail.test', 'corp.io', 'webmail.net']), 'EMAIL'); break;
        case 'PHONE': push('+' + rng.int(9) + rng.int(9) + ' ' + (200 + rng.int(700)) + ' ' + (1000 + rng.int(8999)) + ' ' + (100 + rng.int(899)), 'PHONE'); break;
        case 'NAME': push(rng.pick(PII_NAMES), 'NAME'); break;
        case 'CARD': push((4000 + rng.int(999)) + ' ' + (1000 + rng.int(8999)) + ' ' + (1000 + rng.int(8999)) + ' ' + (1000 + rng.int(8999)), 'CARD'); break;
        case 'KEY': push(rng.pick(['sk', 'pk', 'api', 'ghp']) + '-' + randomToken(rng, 18), 'KEY'); break;
        case 'ADDRESS': push(rng.int(400) + ' ' + rng.pick(PII_STREETS) + ' ' + rng.pick(PII_CITIES), 'ADDRESS'); break;
        case 'DATE': push(2019 + rng.int(7) + '-' + pad2(1 + rng.int(12)) + '-' + pad2(1 + rng.int(28)), 'DATE'); break;
        default: break;
      }
    }
    if (rng.next() < 0.4) push(rng.pick(['thanks', 'regards', 'please advise', 'asap']), 'O');
    texts.push(parts.join(' '));
    tagSeqs.push(tseq);
  }
  const vocab = buildCharVocab(texts, 160);
  const x = new Int32Array(texts.length * cfg.maxLen);
  const y = new Int32Array(texts.length * cfg.maxLen);
  const lengths = new Int32Array(texts.length);
  const meta = [];
  texts.forEach((text, i) => {
    const enc = encodeChars(text, vocab, cfg.maxLen);
    let pos = 0;
    const wordTags = tagSeqs[i];
    const words = text.split(' ');
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi];
      for (let ci = 0; ci < w.length && pos < cfg.maxLen; ci++) {
        y[i * cfg.maxLen + pos] = tagIndex[wordTags[wi]] || 0;
        pos++;
      }
      if (pos < cfg.maxLen) {
        y[i * cfg.maxLen + pos] = tagIndex[wordTags[wi]] || 0;
        pos++;
      }
    }
    for (let k = 0; k < enc.length; k++) x[i * cfg.maxLen + k] = enc.ids[k];
    lengths[i] = Math.min(pos, cfg.maxLen);
    meta.push({ text, preview: text.slice(0, 110), entities: wordTags.filter((t) => t.startsWith('B-')).length });
  });
  const yFloat = new Float32Array(y);
  return createDataset({
    name: 'PII token tagging',
    slug: 'pii',
    description: 'Sentences with inserted email addresses, phone numbers, card numbers, API keys, names, postal addresses and dates. Labels are BIO tags at the character level, which is what a tiny sequence labeller needs to redact text before it is logged.',
    modality: 'text',
    task: 'seq_tag',
    source: 'preset',
    presetKey: 'pii',
    license: Object.assign({}, CC0),
    x: new Float32Array(x),
    xShape: [cfg.maxLen],
    y: yFloat,
    yShape: [cfg.maxLen],
    labels: tags,
    lengths,
    meta,
    featureSpec: { kind: 'char_ids', vocab: vocab.size, maxLen: cfg.maxLen },
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'pii', perClass, seed: cfg.seed, vocab: vocab.size, reference: REFERENCE_DATASETS.pii },
    config: cfg,
    vocab
  });
}

function randomToken(rng, len) {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let s = '';
  for (let i = 0; i < len; i++) s += chars[rng.int(chars.length)];
  return s;
}

function pad2(n) {
  return String(n).padStart(2, '0');
}

const CODE_SNIPPETS = {
  python: [
    'def load_records(path):\n    with open(path) as fh:\n        return [json.loads(line) for line in fh]',
    'import numpy as np\narr = np.zeros((3, 4), dtype=np.float32)\nfor i in range(arr.shape[0]):\n    arr[i] += i * 0.5',
    'class Trainer:\n    def __init__(self, model, lr=1e-3):\n        self.model = model\n        self.lr = lr',
    'result = {k: v for k, v in items.items() if v is not None}\nprint(f"total {len(result)}")',
    'try:\n    value = int(raw)\nexcept ValueError:\n    value = 0\nfinally:\n    log.debug(value)',
    'async def fetch(session, url):\n    async with session.get(url) as resp:\n        return await resp.json()'
  ],
  javascript: [
    'const rows = data.filter(d => d.active).map(d => ({ id: d.id, name: d.name }));\nconsole.log(rows.length);',
    'function debounce(fn, wait) {\n  let t;\n  return (...args) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), wait);\n  };\n}',
    'export default function App({ title }) {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(count + 1)}>{title}</button>;\n}',
    'await Promise.all(urls.map(async u => {\n  const r = await fetch(u);\n  return r.json();\n}));',
    'const obj = { ...base, nested: { ...base.nested, flag: true } };\nObject.freeze(obj);'
  ],
  typescript: [
    'interface Config { readonly host: string; port?: number; }\nconst c: Config = { host: "localhost" };',
    'type Result<T> = { ok: true; value: T } | { ok: false; error: string };\nfunction parse(s: string): Result<number> { return { ok: true, value: Number(s) }; }',
    'export function map<T, U>(xs: T[], f: (x: T) => U): U[] { return xs.map(f); }',
    'class Store<S extends object> { private state: S; constructor(init: S) { this.state = init; } }'
  ],
  rust: [
    'fn main() {\n    let v: Vec<i32> = (0..10).filter(|x| x % 2 == 0).collect();\n    println!("{:?}", v);\n}',
    'impl Default for Config {\n    fn default() -> Self {\n        Self { retries: 3, timeout: Duration::from_secs(5) }\n    }\n}',
    'pub struct Node<T> { value: T, next: Option<Box<Node<T>>> }\nlet n = Node { value: 1, next: None };',
    'match result {\n    Ok(v) => println!("got {}", v),\n    Err(e) => eprintln!("failed: {}", e),\n}'
  ],
  go: [
    'func main() {\n\tvals := []int{1, 2, 3}\n\tfor i, v := range vals {\n\t\tfmt.Println(i, v)\n\t}\n}',
    'if err != nil {\n\treturn fmt.Errorf("load config: %w", err)\n}',
    'type Server struct {\n\tAddr string\n\tmu   sync.Mutex\n}\nfunc (s *Server) Start() error { return nil }',
    'go func() { ch <- compute() }()\nselect {\ncase v := <-ch:\n\tlog.Println(v)\ncase <-time.After(time.Second):\n}'
  ],
  java: [
    'public class Main {\n    public static void main(String[] args) {\n        List<String> names = new ArrayList<>();\n        names.add("a");\n    }\n}',
    '@Override\nprotected void doGet(HttpServletRequest req, HttpServletResponse res) throws IOException {\n    res.getWriter().println("ok");\n}',
    'final Map<String, Integer> counts = new HashMap<>();\ncounts.merge(key, 1, Integer::sum);'
  ],
  csharp: [
    'public class Order {\n    public int Id { get; set; }\n    public decimal Total { get; set; }\n}',
    'var items = await _context.Orders.Where(o => o.Total > 10).ToListAsync();\nConsole.WriteLine(items.Count);',
    'namespace App.Services {\n  public interface ICache { Task<T?> Get<T>(string key); }\n}'
  ],
  c: [
    '#include <stdio.h>\nint main(void) {\n    char buf[64];\n    if (fgets(buf, sizeof buf, stdin) == NULL) return 1;\n    printf("%s", buf);\n    return 0;\n}',
    'void *ptr = malloc(sizeof(int) * n);\nif (ptr == NULL) { perror("malloc"); exit(EXIT_FAILURE); }\nfree(ptr);',
    'static inline int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }'
  ],
  cpp: [
    '#include <vector>\n#include <algorithm>\nint main() {\n    std::vector<int> v{3, 1, 2};\n    std::sort(v.begin(), v.end());\n}',
    'template <typename T>\nclass Ring { public: void push(T&& v) { buf_.emplace_back(std::move(v)); } private: std::vector<T> buf_; };',
    'auto lambda = [&](const auto& x) { return x.size(); };\nstd::cout << lambda(std::string("hi")) << std::endl;'
  ],
  ruby: [
    'def greet(name)\n  puts "hello #{name}"\nend\ngreet("world")',
    'users.each_with_object({}) { |u, h| h[u.id] = u.name }.compact',
    'class Account < ApplicationRecord\n  validates :email, presence: true\n  has_many :orders\nend'
  ],
  php: [
    '<?php\nfunction total(array $items): float {\n    return array_sum(array_column($items, "price"));\n}',
    '$rows = $pdo->query("SELECT id, name FROM users")->fetchAll(PDO::FETCH_ASSOC);\nforeach ($rows as $r) { echo $r["name"]; }'
  ],
  swift: [
    'struct Point { let x: Double; let y: Double }\nlet p = Point(x: 1, y: 2)\nprint(p.x)',
    'func fetch() async throws -> [String] {\n    let (data, _) = try await URLSession.shared.data(from: url)\n    return try JSONDecoder().decode([String].self, from: data)\n}'
  ],
  kotlin: [
    'data class User(val id: Int, val name: String)\nval users = listOf(User(1, "a"), User(2, "b"))\nprintln(users.map { it.name })',
    'suspend fun load(): Result<String> = runCatching { client.get(url).bodyAsText() }'
  ],
  bash: [
    '#!/usr/bin/env bash\nset -euo pipefail\nfor f in *.txt; do\n  wc -l "$f"\ndone',
    'if [[ -z "${VAR:-}" ]]; then\n  echo "missing" >&2\n  exit 1\nfi',
    'curl -sS "$URL" | jq -r \'.items[].name\''
  ],
  sql: [
    'SELECT u.id, COUNT(o.id) AS orders\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nGROUP BY u.id\nHAVING COUNT(o.id) > 3;',
    'WITH recent AS (SELECT * FROM events WHERE ts > NOW() - INTERVAL 7 DAY)\nSELECT date_trunc(\'day\', ts), count(*) FROM recent GROUP BY 1;',
    'CREATE INDEX idx_orders_user ON orders (user_id, created_at DESC);'
  ],
  html: [
    '<div class="card">\n  <h2>Title</h2>\n  <p>Body text</p>\n  <a href="/more">More</a>\n</div>',
    '<form action="/submit" method="post">\n  <label for="email">Email</label>\n  <input id="email" name="email" type="email" required>\n  <button type="submit">Send</button>\n</form>'
  ]
};

export function collectUnique(target, make, sigFn, cap) {
  const seen = new Set();
  const out = [];
  let tries = 0;
  const limit = cap || target * 120 + 200;
  while (out.length < target && tries < limit) {
    tries++;
    const item = make();
    if (!item) continue;
    const sig = sigFn(item);
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push(item);
  }
  return out;
}

const IDENT_SYNONYMS = [
  ['items', 'rows'], ['value', 'result'], ['data', 'payload'], ['count', 'total'],
  ['index', 'pos'], ['name', 'label'], ['list', 'entries'], ['temp', 'scratch'],
  ['config', 'settings'], ['buffer', 'chunk'], ['start', 'begin'], ['limit', 'cap']
];

function mutateCode(text, rng) {
  let t = text;
  const roll = rng.next();
  if (roll < 0.22) {
    const pair = rng.pick(IDENT_SYNONYMS);
    const from = rng.next() < 0.5 ? pair[0] : pair[1];
    const to = from === pair[0] ? pair[1] : pair[0];
    t = t.replace(new RegExp('\\b' + from + '\\b', 'g'), to);
  }
  if (rng.next() < 0.3) {
    t = t.replace(/\b(\d{1,4})\b/g, (m) => (rng.next() < 0.5 ? String(Math.max(0, parseInt(m, 10) + rng.int(9) - 4)) : m));
  }
  if (rng.next() < 0.18) {
    t = t.replace(/"/g, (m) => (rng.next() < 0.5 ? "'" : m));
  }
  const lines = t.split('\n');
  if (rng.next() < 0.2 && lines.length > 2) {
    const i = rng.int(lines.length - 1);
    lines.splice(i, 0, '');
    t = lines.join('\n');
  }
  if (rng.next() < 0.16) {
    t = t.split('\n').map((l) => l.replace(/^( +)/, (m) => (m.length % 4 === 0 ? m + '  ' : m + ' '))).join('\n');
  }
  if (rng.next() < 0.14 && lines.length > 3) {
    const i = 1 + rng.int(lines.length - 2);
    const a = lines[i];
    lines[i] = lines[i - 1];
    lines[i - 1] = a;
    t = lines.join('\n');
  }
  if (rng.next() < 0.1 && lines.length > 2) {
    const i = rng.int(lines.length);
    lines.splice(i, 1);
    t = lines.join('\n');
  }
  return t;
}

export function genCodeLanguage(opts) {
  const cfg = Object.assign({ seed: 55, size: 'medium', dim: 1024, languages: null }, opts || {});
  const rng = new Rng(cfg.seed);
  const langs = cfg.languages || Object.keys(CODE_SNIPPETS);
  const perClass = { small: 22, medium: 55, large: 130 }[cfg.size] || 55;
  const spec = { dim: cfg.dim, charN: [2, 5], wordN: [1, 1], signed: true, norm: 'l2', lowercase: false };
  const rows = [];
  for (const lang of langs) {
    const snippets = CODE_SNIPPETS[lang];
    const made = collectUnique(perClass, () => {
      let text = snippets[rng.int(snippets.length)];
      const lines = text.split('\n');
      if (rng.next() < 0.4 && lines.length > 2) {
        const keep = 2 + rng.int(lines.length - 1);
        const start = rng.int(Math.max(1, lines.length - keep));
        text = lines.slice(start, start + keep).join('\n');
      }
      text = mutateCode(text, rng);
      if (text.length < 16) return null;
      return { text, lang };
    }, (r) => r.text);
    for (const r of made) rows.push(r);
  }
  const shuffled = rng.shuffle(rows);
  const x = new Float32Array(shuffled.length * cfg.dim);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(hashFeatures(r.text, spec), i * cfg.dim);
    y[i] = langs.indexOf(r.lang);
    meta.push({ text: r.text, preview: r.text.replace(/\n/g, ' ⏎ ').slice(0, 110), language: r.lang });
  });
  return createDataset({
    name: 'Code language detection',
    slug: 'code-language',
    description: 'Source snippets across ' + langs.length + ' languages. Features are case-sensitive hashed character n-grams, because braces, keywords and indentation are the actual signal.',
    modality: 'text',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'code',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    labels: langs,
    meta,
    featureSpec: spec,
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'code', languages: langs, perClass, seed: cfg.seed, reference: REFERENCE_DATASETS.code },
    config: cfg
  });
}

const DICT_WORDS = ['password', 'dragon', 'football', 'sunshine', 'princess', 'monkey', 'shadow', 'superman', 'letmein', 'welcome', 'computer', 'internet', 'flower', 'hunter', 'pepper', 'charlie', 'thomas', 'rocket', 'summer', 'winter', 'matrix', 'cookie', 'banana', 'guitar', 'soccer'];
const NAME_WORDS = ['james', 'emma', 'oliver', 'ava', 'noah', 'mia', 'liam', 'sophia', 'lucas', 'isla'];
const KEYBOARD_ROWS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm', '1234567890', 'qazwsx', 'edcrfv'];

export function genPasswordStrength(opts) {
  const cfg = Object.assign({ seed: 66, size: 'medium', dim: 320 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 400, medium: 1100, large: 2800 }[cfg.size] || 1100;
  const spec = { dim: cfg.dim, charN: [1, 4], wordN: [0, 0], signed: true, norm: 'l1', lowercase: false };
  const rows = [];
  const generators = [
    { weight: 0.22, label: 0, make: () => rng.pick(DICT_WORDS) },
    { weight: 0.16, label: 0, make: () => rng.pick(DICT_WORDS) + rng.pick(['123', '1', '01', '!', '1!']) },
    { weight: 0.14, label: 1, make: () => rng.pick(NAME_WORDS) + rng.int(9000 + 999) },
    { weight: 0.1, label: 1, make: () => leetify(rng, rng.pick(DICT_WORDS)) },
    { weight: 0.09, label: 2, make: () => rng.pick(DICT_WORDS) + rng.pick(DICT_WORDS) + rng.int(99) },
    { weight: 0.07, label: 2, make: () => keyboardWalk(rng) },
    { weight: 0.08, label: 3, make: () => passphrase(rng) },
    { weight: 0.06, label: 3, make: () => mixedTokens(rng, 10) },
    { weight: 0.05, label: 4, make: () => mixedTokens(rng, 14) },
    { weight: 0.03, label: 4, make: () => mixedTokens(rng, 20) }
  ];
  const cum = [];
  let acc = 0;
  for (const g of generators) {
    acc += g.weight;
    cum.push(acc);
  }
  const made = collectUnique(total, () => {
    const r = rng.next();
    let gi = 0;
    while (gi < cum.length - 1 && r > cum[gi]) gi++;
    const pw = generators[gi].make();
    const entropy = estimatePasswordBits(pw);
    let label = generators[gi].label;
    if (entropy < 24) label = Math.min(label, 0);
    else if (entropy < 36) label = Math.min(label, 1);
    else if (entropy < 52) label = Math.max(Math.min(label, 2), 1);
    else if (entropy < 72) label = Math.max(Math.min(label, 3), 2);
    else label = Math.max(label, 3);
    return { text: pw, label, bits: entropy };
  }, (r) => r.text);
  for (const r of made) rows.push(r);
  const x = new Float32Array(rows.length * cfg.dim);
  const y = new Float32Array(rows.length);
  const y2 = new Float32Array(rows.length);
  const meta = [];
  rows.forEach((r, i) => {
    x.set(hashFeatures(r.text, spec), i * cfg.dim);
    y[i] = r.label;
    y2[i] = r.bits;
    meta.push({ text: r.text, preview: maskPassword(r.text), bits: r.bits });
  });
  return createDataset({
    name: 'Password strength',
    slug: 'password-strength',
    description: 'Passwords generated from five strategies: dictionary words, dictionary plus suffix, name plus number, leet substitution, keyboard walks, passphrases and random mixed-character strings. The target class comes from an estimated bit-strength, so the model learns structure rather than a symbol-counting rule.',
    modality: 'text',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'password',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    labels: ['very weak', 'weak', 'fair', 'strong', 'very strong'],
    meta,
    featureSpec: spec,
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'password', total, seed: cfg.seed, reference: REFERENCE_DATASETS.password, note: 'No real leaked credentials are used or embedded. Everything here is generated from a seeded PRNG.' },
    config: cfg
  });
}

function maskPassword(pw) {
  if (pw.length <= 2) return '•'.repeat(pw.length);
  return pw[0] + '•'.repeat(Math.max(1, pw.length - 2)) + pw[pw.length - 1];
}

function leetify(rng, word) {
  const map = { a: '4', e: '3', i: '1', o: '0', s: '5', t: '7', b: '8', g: '9' };
  let out = '';
  for (const ch of word) {
    if (rng.next() < 0.45 && map[ch]) out += map[ch];
    else if (rng.next() < 0.15) out += ch.toUpperCase();
    else out += ch;
  }
  return out;
}

function keyboardWalk(rng) {
  const row = rng.pick(KEYBOARD_ROWS);
  const start = rng.int(Math.max(1, row.length - 4));
  const len = 3 + rng.int(5);
  let out = '';
  for (let i = 0; i < len; i++) out += row[(start + i) % row.length];
  if (rng.next() < 0.5) out = out[0].toUpperCase() + out.slice(1);
  return out;
}

function passphrase(rng) {
  const n = 3 + rng.int(3);
  const parts = [];
  for (let i = 0; i < n; i++) parts.push(rng.pick(DICT_WORDS));
  const sep = rng.pick(['-', '_', '.', ' ', '']);
  let out = parts.join(sep);
  if (rng.next() < 0.5) out += rng.int(999);
  return out;
}

function mixedTokens(rng, len) {
  const lower = 'abcdefghijklmnopqrstuvwxyz';
  const upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const digits = '0123456789';
  const symbols = '!@#$%^&*()-_=+[]{};:,.<>?';
  let out = '';
  for (let i = 0; i < len; i++) {
    const r = rng.next();
    if (r < 0.4) out += lower[rng.int(lower.length)];
    else if (r < 0.62) out += upper[rng.int(upper.length)];
    else if (r < 0.85) out += digits[rng.int(digits.length)];
    else out += symbols[rng.int(symbols.length)];
  }
  return out;
}

function estimatePasswordBits(pw) {
  let pool = 0;
  if (/[a-z]/.test(pw)) pool += 26;
  if (/[A-Z]/.test(pw)) pool += 26;
  if (/[0-9]/.test(pw)) pool += 10;
  if (/[^a-zA-Z0-9]/.test(pw)) pool += 26;
  const base = pw.length * Math.log2(Math.max(2, pool));
  let penalty = 0;
  const lower = pw.toLowerCase();
  for (const w of DICT_WORDS) if (lower.includes(w)) penalty += w.length * 2.2;
  for (const w of NAME_WORDS) if (lower.includes(w)) penalty += w.length * 2.0;
  for (const row of KEYBOARD_ROWS) {
    for (let i = 0; i + 3 <= row.length; i++) if (lower.includes(row.slice(i, i + 3))) penalty += 5;
  }
  if (/^(.)\1+$/.test(pw)) penalty += base * 0.8;
  const unique = new Set(pw.split('')).size;
  if (unique < pw.length * 0.5) penalty += base * 0.25;
  return Math.max(2, base - penalty);
}

const PARAPHRASE_PAIRS = [
  ['how do i reset my password', 'what is the process for changing my password'],
  ['where is my order', 'can you tell me the status of my delivery'],
  ['i want a refund', 'give me my money back please'],
  ['what time does the store close', 'when do you shut for the day'],
  ['how much does shipping cost', 'what will delivery add to the price'],
  ['can i change my address', 'is it possible to update the delivery address'],
  ['the app keeps crashing', 'the application closes unexpectedly every time'],
  ['do you ship internationally', 'is overseas delivery available'],
  ['i forgot my username', 'i cannot remember my login name'],
  ['how do i cancel my subscription', 'what are the steps to end my membership'],
  ['is there a student discount', 'do you offer a reduced price for students'],
  ['my card was declined', 'the payment did not go through'],
  ['how long does a refund take', 'when will the money appear back in my account'],
  ['can i speak to a human', 'i would like to talk to a real person'],
  ['what is your return policy', 'how do returns work here'],
  ['the screen is blank', 'nothing shows up on the display'],
  ['how do i update my email', 'i need to change the email on my account'],
  ['is the server down', 'are you having an outage right now'],
  ['how do i track my parcel', 'where can i follow my package'],
  ['i need an invoice', 'please send me a receipt for my records'],
  ['how long will the repair take', 'when will the service be finished'],
  ['i cannot log in to my account', 'my login is not working'],
  ['do you ship to other countries', 'is international delivery available'],
  ['the item arrived damaged', 'my package was broken when it came'],
  ['can i speak to a person', 'i would like to talk to a human agent'],
  ['what are your opening hours', 'when are you open during the week'],
  ['how do i cancel my subscription', 'i want to stop the recurring payment'],
  ['is there a warranty on this', 'does the product come with a guarantee'],
  ['i was charged twice', 'there is a duplicate charge on my card'],
  ['where can i download the manual', 'how do i find the user guide'],
  ['the app keeps crashing', 'my application closes by itself'],
  ['can i return this after thirty days', 'is a late return still possible'],
  ['do you have this in stock', 'is the item available right now'],
  ['how do i update my card details', 'i need to change my payment method'],
  ['the battery drains too fast', 'my device loses charge quickly'],
  ['who is my account manager', 'which representative handles our contract'],
  ['can you send a replacement', 'i need a new unit shipped to me'],
  ['what does this error code mean', 'how should i read the fault message'],
  ['i want to upgrade my plan', 'can i move to a higher tier'],
  ['the invoice has the wrong address', 'my billing details are incorrect on the statement']
];

export function genSentenceEmbeddings(opts) {
  const cfg = Object.assign({ seed: 77, size: 'medium', dim: 512, reps: 14, hardNegatives: true }, opts || {});
  const rng = new Rng(cfg.seed);
  const reps = { small: 5, medium: 11, large: 26 }[cfg.size] || 11;
  const spec = { dim: cfg.dim, charN: [2, 4], wordN: [1, 2], signed: true, norm: 'l2', lowercase: true };
  const xA = [];
  const xB = [];
  const groups = [];
  const meta = [];
  PARAPHRASE_PAIRS.forEach((pair, gi) => {
    const seenPair = new Set();
    let attempts = 0;
    while (seenPair.size < reps && attempts < reps * 60) {
      attempts++;
      const r = seenPair.size;
      const a = r % 2 === 0 ? pair[0] : pair[1];
      const b = r % 2 === 0 ? pair[1] : pair[0];
      const pa = perturbPhrase(a, rng);
      const pb = perturbPhrase(b, rng);
      const sig = pa + '||' + pb;
      if (seenPair.has(sig)) continue;
      seenPair.add(sig);
      xA.push(pa);
      xB.push(pb);
      groups.push(gi);
      meta.push({ text: pa + '  ||  ' + pb, preview: (pa + ' → ' + pb).slice(0, 110), group: gi });
    }
  });
  const n = xA.length;
  const x = new Float32Array(n * cfg.dim);
  const x2 = new Float32Array(n * cfg.dim);
  const y = new Float32Array(n);
  const groupArr = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    x.set(hashFeatures(xA[i], spec), i * cfg.dim);
    x2.set(hashFeatures(xB[i], spec), i * cfg.dim);
    y[i] = 1;
    groupArr[i] = groups[i];
  }
  return createDataset({
    name: 'Paraphrase pairs for embeddings',
    slug: 'sentence-embeddings',
    description: PARAPHRASE_PAIRS.length + ' meaning groups, each expanded into ' + reps + ' surface variations. The model is trained with InfoNCE so that paraphrases pull together and everything else pushes apart. Group ids keep the split honest.',
    modality: 'text',
    task: 'embedding',
    source: 'preset',
    presetKey: 'embeddings',
    license: Object.assign({}, CC0),
    x,
    x2,
    xShape: [cfg.dim],
    y,
    yShape: [1],
    groups: groupArr,
    labels: ['pair'],
    meta,
    featureSpec: spec,
    splitMethod: 'group',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'embeddings', groups: PARAPHRASE_PAIRS.length, reps, seed: cfg.seed, reference: REFERENCE_DATASETS.embeddings },
    config: cfg
  });
}

function perturbPhrase(text, rng) {
  let t = text;
  if (rng.next() < 0.45) t = rng.pick(['hey ', 'so ', 'quick question ', 'hi ', 'hello ', 'sorry to bother you but ', 'one thing ', '']) + t;
  if (rng.next() < 0.35) t = t + rng.pick([' thanks', ' please', '?', ' asap', ' ta', ' if that is ok', ' when you get a chance', ' cheers', '']);
  if (rng.next() < 0.2) t = t.replace(/\bmy\b/, rng.pick(['the', 'our', 'this'])).replace(/\bi\b/, rng.pick(['we', 'i']));
  if (rng.next() < 0.18) t = applyTypo(t, rng, 0.05);
  if (rng.next() < 0.15) {
    const words = t.split(' ');
    if (words.length > 3) {
      const i = rng.int(words.length - 2);
      const tmp = words[i];
      words[i] = words[i + 1];
      words[i + 1] = tmp;
      t = words.join(' ');
    }
  }
  return t;
}

const LM_CORPUS = [
  'git commit -m "fix the failing test in the auth module"',
  'git push origin main and open a pull request for review',
  'npm install then npm run build and serve the output folder',
  'docker compose up -d and tail the logs for the api service',
  'kubectl get pods -n production and describe the failing one',
  'the build succeeded but the smoke test timed out after sixty seconds',
  'run the migration locally before deploying to the staging cluster',
  'check the response headers for the cache control directive',
  'set the environment variable and restart the worker process',
  'grep the logs for the request id and print the surrounding lines',
  'the model trained for twelve epochs and validation loss stopped improving',
  'export the weights to onnx and compare accuracy before and after quantisation',
  'we shipped the feature behind a flag and rolled it out to five percent',
  'the regression came from a change in the serialisation layer',
  'please add a unit test that covers the empty input case',
  'the latency budget is forty milliseconds end to end including the network',
  'shuffle the training set with a fixed seed so the run is reproducible',
  'the dataset hash is recorded with every run for auditability',
  'reduce the learning rate when the validation metric plateaus',
  'the tiny model fits in a few hundred kilobytes and runs on the device'
];

export function genAutocomplete(opts) {
  const cfg = Object.assign({ seed: 88, size: 'medium', seqLen: 64 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 260, medium: 760, large: 1900 }[cfg.size] || 760;
  const corpus = LM_CORPUS.join(' \n ');
  const vocab = buildCharVocab([corpus], 128);
  const x = new Float32Array(total * cfg.seqLen);
  const y = new Float32Array(total);
  const lengths = new Int32Array(total);
  const meta = [];
  let pos = 0;
  const lmTexts = collectUnique(total, () => {
    const start = rng.int(Math.max(1, corpus.length - cfg.seqLen));
    let text = corpus.slice(start, start + cfg.seqLen);
    if (rng.next() < 0.12) text = applyTypo(text, rng, 0.02);
    return text.length >= cfg.seqLen - 4 ? text : null;
  }, (t) => t, total * 200);
  for (let i = 0; i < lmTexts.length; i++) {
    const text = lmTexts[i];
    const enc = encodeChars(text, vocab, cfg.seqLen);
    for (let k = 0; k < enc.ids.length; k++) x[i * cfg.seqLen + k] = enc.ids[k];
    lengths[i] = enc.length;
    y[i] = enc.length;
    meta.push({ text, preview: text.slice(0, 90) });
    pos += enc.length;
  }
  return createDataset({
    name: 'Command and prose autocomplete corpus',
    slug: 'autocomplete',
    task: 'lm',
    description: 'Character-level corpus of shell commands, engineering notes and product sentences. Training objective is next-character prediction; evaluation reports perplexity, top-1 accuracy and bits per character.',
    modality: 'text',
    source: 'preset',
    presetKey: 'autocomplete',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.seqLen],
    y,
    yShape: [1],
    lengths,
    labels: [],
    meta,
    featureSpec: { kind: 'char_ids', vocab: vocab.size, maxLen: cfg.seqLen },
    splitMethod: 'random',
    splitRatios: [0.8, 0.1, 0.1],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'autocomplete', total, seed: cfg.seed, vocab: vocab.size, reference: REFERENCE_DATASETS.lm },
    config: cfg,
    vocab
  });
}

export function fitClip(clip, n, rng, maxOffset) {
  const out = new Float32Array(n);
  const room = Math.max(0, n - clip.length);
  const off = maxOffset === 0 ? 0 : rng.int(Math.min(room, maxOffset) + 1);
  for (let i = 0; i < clip.length && off + i < n; i++) out[off + i] = clip[i];
  return out;
}

export function standardizeDataset(ds, fields) {
  fields = fields || ['x'];
  const per = shapeSize(ds.xShape);
  const idx = ds.splits.train.length ? ds.splits.train : Int32Array.from(ds.x.keys());
  for (const field of fields) {
    const arr = ds[field];
    if (!arr) continue;
    const mean = new Float32Array(per);
    const std = new Float32Array(per);
    for (const i of idx) {
      for (let k = 0; k < per; k++) mean[k] += arr[i * per + k];
    }
    for (let k = 0; k < per; k++) mean[k] /= idx.length;
    for (const i of idx) {
      for (let k = 0; k < per; k++) {
        const d = arr[i * per + k] - mean[k];
        std[k] += d * d;
      }
    }
    for (let k = 0; k < per; k++) std[k] = Math.sqrt(std[k] / idx.length) + 1e-6;
    for (let i = 0; i < ds.count; i++) {
      for (let k = 0; k < per; k++) arr[i * per + k] = (arr[i * per + k] - mean[k]) / std[k];
    }
    ds.featureSpec = Object.assign({}, ds.featureSpec, { [field + 'Mean']: Array.from(mean), [field + 'Std']: Array.from(std) });
  }
  ds.digest = null;
  return ds;
}

function audioFeature(signal, cfg) {
  const mel = logMelFrames(signal, {
    sampleRate: cfg.sampleRate,
    frameSize: cfg.frameSize,
    hop: cfg.hop,
    numMel: cfg.numMel
  });
  const norm = normalizeFeatures(mel.data, mel.frames, mel.bins);
  return { data: norm.data, frames: mel.frames, bins: mel.bins };
}

const AUDIO_FEATURE_DEFAULTS = { sampleRate: 8000, frameSize: 400, hop: 160, numMel: 32, duration: 0.75 };

export function genWakeWord(opts) {
  const cfg = Object.assign({ seed: 99, size: 'medium', snrMin: 3, snrMax: 22, noiseKinds: ['white', 'pink', 'babble', 'hum'] }, opts || {});
  const rng = new Rng(cfg.seed);
  const counts = { small: [70, 90], medium: [180, 230], large: [430, 540] }[cfg.size] || [180, 230];
  const F = AUDIO_FEATURE_DEFAULTS;
  const n = Math.round(F.sampleRate * F.duration);
  const samples = [];
  for (let i = 0; i < counts[0]; i++) {
    const phrase = WAKE_PHRASES[rng.int(WAKE_PHRASES.length)];
    const clip = synthKeyword(F.sampleRate, F.duration * rng.range(0.82, 1.0), rng, phrase);
    samples.push({ clean: fitClip(peakNormalize(clip), n, rng, Math.round(n * 0.12)), label: 1, name: phrase.name });
  }
  for (let i = 0; i < counts[1]; i++) {
    const r = rng.next();
    if (r < 0.62) {
      const phrase = NEGATIVE_PHRASES[rng.int(NEGATIVE_PHRASES.length)];
      const clip = synthKeyword(F.sampleRate, F.duration * rng.range(0.75, 1.05), rng, phrase);
      samples.push({ clean: fitClip(peakNormalize(clip), n, rng, Math.round(n * 0.12)), label: 0, name: phrase.name });
    } else if (r < 0.8) {
      const kind = rng.pick(['speech', 'music', 'dog_bark', 'knock', 'doorbell']);
      const clip = soundEvent(rng, kind, F.sampleRate, F.duration);
      samples.push({ clean: fitClip(clip, n, rng, Math.round(n * 0.1)), label: 0, name: kind });
    } else {
      samples.push({ clean: new Float32Array(n), label: 0, name: 'silence' });
    }
  }
  const shuffled = rng.shuffle(samples);
  const feats = shuffled.map((s) => {
    const kind = rng.pick(cfg.noiseKinds);
    const noise = synthNoise(F.sampleRate, F.duration, kind, rng);
    const snr = s.label ? rng.range(cfg.snrMin, cfg.snrMax) : rng.range(cfg.snrMin - 4, cfg.snrMax + 6);
    const noisy = addNoise(s.clean, snr, rng, noise);
    return { feat: audioFeature(noisy, F), label: s.label, name: s.name + ' · ' + kind + ' ' + snr.toFixed(0) + 'dB' };
  });
  const frames = feats[0].feat.frames;
  const bins = feats[0].feat.bins;
  const x = new Float32Array(feats.length * frames * bins);
  const y = new Float32Array(feats.length);
  const meta = [];
  feats.forEach((f, i) => {
    x.set(f.feat.data.subarray(0, frames * bins), i * frames * bins);
    y[i] = f.label;
    meta.push({ preview: f.name + (f.label ? ' (keyword)' : ''), text: f.name, label: f.label });
  });
  const ds = createDataset({
    name: 'Wake word clips',
    slug: 'wake-word',
    description: 'Keyword clips against near-miss phrases, household sounds and silence, each mixed with white, pink, babble or mains-hum noise between ' + cfg.snrMin + ' and ' + cfg.snrMax + ' dB SNR. Features are 32-band log-mel frames.',
    modality: 'audio',
    task: 'binary',
    source: 'preset',
    presetKey: 'wake',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, bins],
    y,
    yShape: [1],
    labels: ['negative', 'keyword'],
    meta,
    featureSpec: { kind: 'log_mel', frames, bins, sampleRate: F.sampleRate, frameSize: F.frameSize, hop: F.hop, numMel: F.numMel },
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    sampleRate: F.sampleRate,
    provenance: { generator: 'wake', positives: counts[0], negatives: counts[1], seed: cfg.seed, reference: REFERENCE_DATASETS.wake },
    config: cfg
  });
  return ds;
}

export function genVad(opts) {
  const cfg = Object.assign({ seed: 111, size: 'medium' }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 90, medium: 220, large: 520 }[cfg.size] || 220;
  const F = { sampleRate: 8000, frameSize: 320, hop: 160, numMel: 32, duration: 0.64 };
  const frames = Math.floor((F.sampleRate * F.duration - F.frameSize) / F.hop) + 1;
  const x = new Float32Array(total * frames * F.numMel);
  const y = new Float32Array(total * frames);
  const meta = [];
  const signals = [];
  for (let i = 0; i < total; i++) {
    const n = Math.round(F.sampleRate * F.duration);
    const speechLen = Math.round(n * rng.range(0.25, 0.9));
    const start = rng.int(Math.max(1, n - speechLen));
    const mode = rng.next();
    const clean = new Float32Array(n);
    if (mode < 0.82) {
      const speech = synthSpeechLike(F.sampleRate, F.duration, rng, { syllableRate: rng.range(3, 6.5) });
      for (let k = 0; k < speechLen && start + k < n; k++) clean[start + k] = speech[k];
    } else {
      const ev = soundEvent(rng, rng.pick(['dog_bark', 'knock', 'doorbell', 'glass_break']), F.sampleRate, F.duration);
      for (let k = 0; k < n; k++) clean[k] = ev[k] * 0.5;
    }
    const noise = synthNoise(F.sampleRate, F.duration, rng.pick(['white', 'pink', 'hum']), rng);
    const noisy = addNoise(clean, rng.range(0, 20), rng, noise);
    const frameLabels = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
      const centre = f * F.hop + F.frameSize / 2;
      frameLabels[f] = mode < 0.82 && centre >= start && centre < start + speechLen ? 1 : 0;
    }
    const feat = audioFeature(noisy, F);
    x.set(feat.data.subarray(0, frames * F.numMel), i * frames * F.numMel);
    y.set(frameLabels, i * frames);
    meta.push({ preview: (mode < 0.82 ? 'speech' : 'event') + ' · ' + Math.round(100 * frameLabels.reduce((a, b) => a + b, 0) / frames) + '% voiced', text: '', speechFraction: frameLabels.reduce((a, b) => a + b, 0) / frames });
    signals.push({ clean, noisy, frameLabels });
  }
  const ds = createDataset({
    name: 'Voice activity frames',
    slug: 'vad',
    description: 'Per-frame speech / non-speech labels over 32-band log-mel frames. Clip onsets are randomised so the model cannot use position as a shortcut, and non-speech events are included to keep false accepts honest.',
    modality: 'audio',
    task: 'frame_binary',
    source: 'preset',
    presetKey: 'vad',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, F.numMel],
    y,
    yShape: [frames],
    labels: ['non-speech', 'speech'],
    meta,
    featureSpec: { kind: 'log_mel', frames, bins: F.numMel, sampleRate: F.sampleRate, frameSize: F.frameSize, hop: F.hop, numMel: F.numMel },
    splitMethod: 'random',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    sampleRate: F.sampleRate,
    provenance: { generator: 'vad', total, seed: cfg.seed, reference: REFERENCE_DATASETS.vad },
    config: Object.assign({}, cfg, { frameRate: F.sampleRate / F.hop })
  });
  return ds;
}

export function genNoiseSuppression(opts) {
  const cfg = Object.assign({ seed: 122, size: 'medium', snrMin: -3, snrMax: 12 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 60, medium: 140, large: 320 }[cfg.size] || 140;
  const sampleRate = 8000;
  const duration = 0.5;
  const frameSize = 256;
  const hop = 128;
  const n = Math.round(sampleRate * duration);
  const frames = Math.floor((n - frameSize) / hop) + 1;
  const bins = frameSize / 2 + 1;
  const x = new Float32Array(total * frames * bins);
  const y = new Float32Array(total * frames * bins);
  const wave = [];
  const meta = [];
  for (let i = 0; i < total; i++) {
    const clean = synthSpeechLike(sampleRate, duration, rng, { syllableRate: rng.range(3.5, 6) });
    const kind = rng.pick(['white', 'pink', 'babble', 'hum', 'white', 'babble']);
    const noise = synthNoise(sampleRate, duration, kind, rng);
    const snr = rng.range(cfg.snrMin, cfg.snrMax);
    const noisy = addNoise(clean, snr, rng, noise);
    const cleanSpec = linearPowerFrames(clean, { frameSize, hop });
    const noisySpec = linearPowerFrames(noisy, { frameSize, hop });
    const f = Math.min(frames, cleanSpec.frames, noisySpec.frames);
    const cleanLog = new Float32Array(f * bins);
    const mask = new Float32Array(f * bins);
    for (let k = 0; k < f * bins; k++) {
      const cp = cleanSpec.mag[k] * cleanSpec.mag[k];
      const np2 = noisySpec.mag[k] * noisySpec.mag[k];
      cleanLog[k] = Math.log(Math.max(1e-8, cp));
      mask[k] = Math.max(0, Math.min(1, cleanSpec.mag[k] / Math.max(1e-6, noisySpec.mag[k])));
    }
    const noisyLog = new Float32Array(f * bins);
    for (let k = 0; k < f * bins; k++) {
      noisyLog[k] = Math.log(Math.max(1e-8, noisySpec.mag[k] * noisySpec.mag[k]));
    }
    const xRow = new Float32Array(frames * bins);
    const yRow = new Float32Array(frames * bins);
    xRow.set(noisyLog.subarray(0, f * bins));
    yRow.set(mask.subarray(0, f * bins));
    x.set(xRow, i * frames * bins);
    y.set(yRow, i * frames * bins);
    wave.push({ clean: new Float32Array(clean), noisy: new Float32Array(noisy), phase: noisySpec.phase, mag: noisySpec.mag, frames: noisySpec.frames, bins: noisySpec.bins, frameSize, hop, sampleRate, snr });
    meta.push({ preview: kind + ' noise at ' + snr.toFixed(1) + ' dB SNR', text: '', snr, noiseKind: kind });
  }
  const ds = createDataset({
    name: 'Speech noise suppression pairs',
    slug: 'noise-suppression',
    description: 'Clean / noisy speech pairs. The input is the log power spectrum of the noisy signal and the target is the ideal magnitude mask, clean magnitude over noisy magnitude, so multiplying it back onto the noisy spectrum recovers the clean one. Raw waveforms and noisy phase are kept so SI-SDR, segmental SNR and the perceptual estimate can be measured after reconstruction.',
    modality: 'audio',
    task: 'spectral_mask',
    source: 'preset',
    presetKey: 'denoise',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, bins],
    y,
    yShape: [frames, bins],
    wave,
    labels: [],
    meta,
    featureSpec: { kind: 'log_power_spectrum', frames, bins, sampleRate, frameSize, hop },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    sampleRate,
    provenance: { generator: 'noise_suppression', total, seed: cfg.seed, reference: REFERENCE_DATASETS.denoise },
    config: Object.assign({}, cfg, { frameRate: sampleRate / hop, sampleRate })
  });
  return ds;
}

const PITCH_NOTES = (() => {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const out = [];
  for (let i = 0; i < 36; i++) {
    const midi = 43 + i;
    out.push({ midi, hz: 440 * Math.pow(2, (midi - 69) / 12), name: names[midi % 12] + (Math.floor(midi / 12) - 1) });
  }
  return out;
})();

export function genPitch(opts) {
  const cfg = Object.assign({ seed: 133, size: 'medium', harmonics: 6 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 180, medium: 460, large: 1000 }[cfg.size] || 460;
  const sampleRate = 16000;
  const duration = 0.32;
  const frameSize = 2048;
  const hop = 512;
  const n = Math.round(sampleRate * duration);
  const frames = Math.floor((n - frameSize) / hop) + 1;
  const binWidth = sampleRate / frameSize;
  const loBin = Math.max(1, Math.floor(60 / binWidth));
  const hiBin = Math.min(frameSize / 2, Math.ceil(1400 / binWidth));
  const numBins = hiBin - loBin + 1;
  const x = new Float32Array(total * frames * numBins);
  const y = new Float32Array(total);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const cls = rng.int(PITCH_NOTES.length);
    const f0 = PITCH_NOTES[cls].hz * Math.pow(2, rng.range(-0.02, 0.02));
    const signal = new Float32Array(n);
    const vibrato = rng.next() < 0.5 ? rng.range(0.002, 0.014) : 0;
    const vRate = rng.range(4, 7);
    const harmonicCount = 1 + rng.int(cfg.harmonics);
    const amps = [];
    for (let h = 1; h <= harmonicCount; h++) amps.push(Math.pow(0.72, h - 1) * rng.range(0.7, 1.3));
    for (let k = 0; k < n; k++) {
      const t = k / sampleRate;
      const f = f0 * (1 + vibrato * Math.sin(2 * Math.PI * vRate * t));
      let v = 0;
      for (let h = 1; h <= harmonicCount; h++) {
        if (f * h > sampleRate / 2) break;
        v += amps[h - 1] * Math.sin(2 * Math.PI * f * h * t + rng.range(0, 0.02));
      }
      const env = Math.min(1, k / (sampleRate * 0.02)) * Math.min(1, (n - k) / (sampleRate * 0.04));
      signal[k] = v * env * 0.35;
    }
    const noisy = addNoise(signal, rng.range(12, 34), rng, synthNoise(sampleRate, duration, rng.pick(['white', 'pink']), rng));
    const spec = linearPowerFrames(noisy, { frameSize, hop });
    const row = new Float32Array(frames * numBins);
    const fUse = Math.min(frames, spec.frames);
    for (let f = 0; f < fUse; f++) {
      for (let b = 0; b < numBins; b++) {
        row[f * numBins + b] = Math.log(Math.max(1e-9, spec.mag[f * spec.bins + loBin + b]));
      }
    }
    let sum = 0;
    for (let k = 0; k < frames * numBins; k++) sum += row[k];
    const mu = sum / (frames * numBins);
    let varSum = 0;
    for (let k = 0; k < frames * numBins; k++) varSum += (row[k] - mu) * (row[k] - mu);
    const sd = Math.sqrt(varSum / (frames * numBins)) + 1e-6;
    for (let k = 0; k < frames * numBins; k++) row[k] = (row[k] - mu) / sd;
    x.set(row.subarray(0, frames * numBins), i * frames * numBins);
    y[i] = cls;
    meta.push({ preview: PITCH_NOTES[cls].name + ' · ' + f0.toFixed(1) + ' Hz · ' + harmonicCount + ' harmonics', text: '', hz: f0, note: PITCH_NOTES[cls].name, midi: PITCH_NOTES[cls].midi });
  }
  const ds = createDataset({
    name: 'Pitch detection frames',
    slug: 'pitch',
    description: 'Synthesised tones from ' + PITCH_NOTES[0].hz.toFixed(0) + ' to ' + PITCH_NOTES[PITCH_NOTES.length - 1].hz.toFixed(0) + ' Hz with 1-6 harmonics, vibrato and additive noise. Features are a 2048-point log magnitude spectrum restricted to ' + loBin + '-' + hiBin + ' bins (60-1400 Hz) so individual harmonics stay resolvable; mel bands would smear them. Each clip is scaled by its own global mean and deviation, never per bin, because per-bin standardisation would erase the spectral shape that carries the pitch. The target is a 36-bin chromatic pitch class, which makes the metric a cent error rather than a raw frequency regression.',
    modality: 'audio',
    task: 'pitch_class',
    source: 'preset',
    presetKey: 'pitch',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, numBins],
    y,
    yShape: [1],
    labels: PITCH_NOTES.map((p) => p.name),
    meta,
    featureSpec: { kind: 'log_linear_spectrum', frames, bins: numBins, sampleRate, frameSize, hop, loBin, hiBin, binWidth, noteTable: PITCH_NOTES },
    splitMethod: 'stratified',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    sampleRate,
    provenance: { generator: 'pitch', total, seed: cfg.seed, reference: REFERENCE_DATASETS.pitch },
    config: Object.assign({}, cfg, { noteTable: PITCH_NOTES })
  });
  return ds;
}

export const SOUND_EVENT_CLASSES = ['doorbell', 'smoke_alarm', 'glass_break', 'baby_cry', 'dog_bark', 'knock', 'siren', 'phone_ring', 'microwave_beep', 'speech', 'music', 'silence'];

export function genSoundEvents(opts) {
  const cfg = Object.assign({ seed: 144, size: 'medium', classes: null, snrMin: 4, snrMax: 26 }, opts || {});
  const rng = new Rng(cfg.seed);
  const classes = cfg.classes || SOUND_EVENT_CLASSES;
  const perClass = { small: 26, medium: 62, large: 150 }[cfg.size] || 62;
  const F = { sampleRate: 8000, frameSize: 400, hop: 160, numMel: 32, duration: 0.8 };
  const frames = Math.floor((F.sampleRate * F.duration - F.frameSize) / F.hop) + 1;
  const rows = [];
  classes.forEach((cls) => {
    for (let i = 0; i < perClass; i++) {
      const n = Math.round(F.sampleRate * F.duration);
      let clean;
      if (cls === 'silence') clean = new Float32Array(n);
      else if (cls === 'speech') clean = synthSpeechLike(F.sampleRate, F.duration, rng, {});
      else clean = soundEvent(rng, cls, F.sampleRate, F.duration);
      const shifted = fitClip(clean, n, rng, Math.round(n * 0.12));
      const noise = synthNoise(F.sampleRate, F.duration, rng.pick(['white', 'pink', 'hum', 'babble']), rng);
      const noisy = cls === 'silence' ? noise : addNoise(shifted, rng.range(cfg.snrMin, cfg.snrMax), rng, noise);
      rows.push({ feat: audioFeature(noisy, F), cls });
    }
  });
  const shuffled = rng.shuffle(rows);
  const x = new Float32Array(shuffled.length * frames * F.numMel);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(r.feat.data.subarray(0, frames * F.numMel), i * frames * F.numMel);
    y[i] = classes.indexOf(r.cls);
    meta.push({ preview: r.cls, text: r.cls });
  });
  return createDataset({
    name: 'Sound event detection',
    slug: 'sound-events',
    description: classes.length + ' event classes synthesised from oscillators, filtered noise bursts and formant voices, mixed into four background noise types. This is the shape of an accessibility or smart-home detector: short clips, fixed vocabulary, on-device.',
    modality: 'audio',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'events',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, F.numMel],
    y,
    yShape: [1],
    labels: classes,
    meta,
    featureSpec: { kind: 'log_mel', frames, bins: F.numMel, sampleRate: F.sampleRate, frameSize: F.frameSize, hop: F.hop, numMel: F.numMel },
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    sampleRate: F.sampleRate,
    provenance: { generator: 'sound_events', classes, perClass, seed: cfg.seed, reference: REFERENCE_DATASETS.events },
    config: cfg
  });
}

export function genBeatOnset(opts) {
  const cfg = Object.assign({ seed: 155, size: 'medium', bpmRange: [70, 160] }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 70, medium: 170, large: 400 }[cfg.size] || 170;
  const sampleRate = 8000;
  const duration = 2.0;
  const frameSize = 400;
  const hop = 160;
  const numMel = 32;
  const frames = Math.floor((sampleRate * duration - frameSize) / hop) + 1;
  const frameRate = sampleRate / hop;
  const x = new Float32Array(total * frames * numMel);
  const y = new Float32Array(total * frames);
  const meta = [];
  const extras = [];
  for (let i = 0; i < total; i++) {
    const bpm = rng.range(cfg.bpmRange[0], cfg.bpmRange[1]);
    const track = rhythmTrack(rng, sampleRate, duration, bpm);
    let signal = track.signal;
    if (rng.next() < 0.55) {
      const pad = soundEvent(rng, rng.pick(['music', 'speech', 'silence']), sampleRate, duration);
      signal = mixSignals([signal, pad], [1, rng.range(0.05, 0.4)]);
    }
    if (rng.next() < 0.7) signal = addNoise(signal, rng.range(8, 28), rng, synthNoise(sampleRate, duration, rng.pick(['white', 'pink', 'hum']), rng));
    const feat = audioFeature(signal, { sampleRate, frameSize, hop, numMel, duration });
    const labels = new Float32Array(frames);
    for (const t of track.onsets) {
      const f = Math.round(t * frameRate);
      for (let k = -1; k <= 1; k++) {
        const fi = f + k;
        if (fi >= 0 && fi < frames) labels[fi] = k === 0 ? 1 : 0.6;
      }
    }
    x.set(feat.data.subarray(0, frames * numMel), i * frames * numMel);
    y.set(labels, i * frames);
    extras.push({ bpm, onsets: track.onsets });
    meta.push({ preview: bpm.toFixed(1) + ' BPM · ' + track.onsets.length + ' onsets', text: '', bpm, onsets: track.onsets.length });
  }
  return createDataset({
    name: 'Beat and onset detection',
    slug: 'beat-onset',
    description: 'Rhythmic tracks between ' + cfg.bpmRange[0] + ' and ' + cfg.bpmRange[1] + ' BPM with transient hits, optionally layered over music or speech and noise. Labels are per-frame onset indicators with one frame of soft tolerance either side.',
    modality: 'audio',
    task: 'onset',
    source: 'preset',
    presetKey: 'onset',
    license: Object.assign({}, CC0),
    x,
    xShape: [frames, numMel],
    y,
    yShape: [frames],
    labels: ['no onset', 'onset'],
    meta,
    featureSpec: { kind: 'log_mel', frames, bins: numMel, sampleRate, frameSize, hop, numMel },
    splitMethod: 'random',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    sampleRate,
    provenance: { generator: 'beat_onset', total, seed: cfg.seed, reference: REFERENCE_DATASETS.onset },
    config: Object.assign({}, cfg, { frameRate, extras })
  });
}

function toUnit(img) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = clamp01(img[i]);
  return out;
}

function addSharpDetail(img, C, H, W, rng) {
  const shapes = 3 + rng.int(5);
  for (let k = 0; k < shapes; k++) {
    const col = [rng.range(0.02, 0.98), rng.range(0.02, 0.98), rng.range(0.02, 0.98)];
    const kind = rng.int(3);
    if (kind === 0) {
      const n = 3 + rng.int(4);
      const cx = rng.range(0.15, 0.85) * W;
      const cy = rng.range(0.15, 0.85) * H;
      const r = rng.range(0.07, 0.24) * Math.min(H, W);
      const pts = [];
      for (let j = 0; j < n; j++) {
        const a = (j / n) * Math.PI * 2 + rng.range(-0.25, 0.25);
        pts.push(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      }
      fillPolygon(img, C, H, W, pts, col);
    } else if (kind === 1) {
      drawLine(img, C, H, W, rng.range(0, W), rng.range(0, H), rng.range(0, W), rng.range(0, H), col, 1 + rng.range(0, 1.6));
    } else {
      const w = rng.range(0.1, 0.34) * W;
      const h = rng.range(0.1, 0.34) * H;
      fillRect(img, C, H, W, Math.round(rng.range(0, W - w)), Math.round(rng.range(0, H - h)), Math.round(rng.range(0, W - w) + w), Math.round(rng.range(0, H - h) + h), col);
    }
  }
  const hf = valueNoise(rng, W, H, Math.max(6, Math.round(W / 2)), 1, 0.5);
  const gain = rng.range(0.1, 0.22);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = hf[y * W + x] * gain;
      for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(img[(c * H + y) * W + x] + v);
    }
  }
}

export function genSuperResolution(opts) {
  const cfg = Object.assign({ seed: 166, size: 'medium', factor: 2, hrSize: 32, patchSize: 16 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 80, medium: 240, large: 560 }[cfg.size] || 240;
  const C = 3;
  const H = cfg.hrSize;
  const W = cfg.hrSize;
  const P = Math.max(4, Math.min(H, cfg.patchSize));
  const perImage = Math.floor(H / P) * Math.floor(W / P);
  const images = Math.max(1, Math.round(total / perImage));
  const count = images * perImage;
  const lr = Math.max(4, Math.round(H / cfg.factor));
  const x = new Float32Array(count * C * P * P);
  const y = new Float32Array(count * C * P * P);
  const meta = [];
  let row = 0;
  for (let i = 0; i < images; i++) {
    const hr = toUnit(synthNatural(rng, C, H, W));
    addSharpDetail(hr, C, H, W, rng);
    const low = areaDownsample(hr, C, H, W, cfg.factor);
    const up = bicubicResize(low, C, lr, lr, H, W);
    for (let k = 0; k < up.length; k++) up[k] = clamp01(up[k]);
    for (let by = 0; by + P <= H; by += P) {
      for (let bx = 0; bx + P <= W; bx += P) {
        for (let c = 0; c < C; c++) {
          for (let py = 0; py < P; py++) {
            for (let px = 0; px < P; px++) {
              const src = (c * H + by + py) * W + bx + px;
              const dst = (c * P + py) * P + px;
              x[row * C * P * P + dst] = up[src];
              y[row * C * P * P + dst] = hr[src];
            }
          }
        }
        meta.push({ preview: cfg.factor + 'x bicubic input · ' + P + '×' + P + ' patch of a ' + H + '×' + W + ' target', text: '' });
        row++;
      }
    }
  }
  return createDataset({
    name: cfg.factor + '× super-resolution patches',
    slug: 'super-resolution',
    description: images + ' procedurally generated ' + H + '×' + W + ' images carrying sharp polygons, lines and high-frequency grain, downsampled ' + cfg.factor + '× by area averaging, brought back up with bicubic interpolation and cut into ' + count + ' non-overlapping ' + P + '×' + P + ' patches. Patch training is what makes a small super-resolution net trainable in a browser: four times the samples at a quarter of the convolution cost. The bicubic result is both the model input and the baseline the PSNR gain is measured against, so a positive gain means the network recovered edge detail that interpolation cannot.',
    modality: 'image',
    task: 'image_map',
    source: 'preset',
    presetKey: 'superres',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, P, P],
    y,
    yShape: [C, P, P],
    labels: [],
    meta,
    featureSpec: { kind: 'rgb_unit', factor: cfg.factor, patchSize: P, sourceSize: H },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'super_resolution', total: count, sourceImages: images, factor: cfg.factor, seed: cfg.seed, reference: REFERENCE_DATASETS.superres },
    config: Object.assign({}, cfg, { peak: 1, patchSize: P, hrSize: H })
  });
}

export function genImageDenoise(opts) {
  const cfg = Object.assign({ seed: 177, size: 'medium', degradation: 'mixed', noiseStd: 0.16, size_ : 32 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 70, medium: 170, large: 400 }[cfg.size] || 170;
  const C = 3;
  const H = cfg.size_ || 32;
  const W = H;
  const x = new Float32Array(total * C * H * W);
  const y = new Float32Array(total * C * H * W);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const clean = toUnit(synthNatural(rng, C, H, W));
    const mode = cfg.degradation === 'mixed' ? rng.pick(['noise', 'jpeg', 'noise_jpeg', 'lowlight']) : cfg.degradation;
    let degraded = clean;
    if (mode === 'noise' || mode === 'noise_jpeg' || mode === 'lowlight') {
      const std = cfg.noiseStd * (mode === 'lowlight' ? rng.range(1.2, 2.2) : rng.range(0.6, 1.3));
      degraded = addGaussianNoise(degraded, rng, std);
    }
    if (mode === 'jpeg' || mode === 'noise_jpeg') {
      degraded = jpegQuantize(degraded, C, H, W, rng.range(0.15, 0.5));
    }
    if (mode === 'lowlight') {
      const g = rng.range(0.35, 0.6);
      for (let k = 0; k < degraded.length; k++) degraded[k] = clamp01(degraded[k] * g + 0.1);
    }
    for (let k = 0; k < degraded.length; k++) degraded[k] = clamp01(degraded[k]);
    x.set(degraded, i * C * H * W);
    y.set(clean, i * C * H * W);
    meta.push({ preview: mode + ' degradation', text: '', mode });
  }
  return createDataset({
    name: 'Image denoising and artefact removal',
    slug: 'image-denoise',
    description: 'Clean procedural images paired with degraded versions: additive Gaussian noise, posterisation that mimics low-quality JPEG, and low-light gain reduction. Input and target are both RGB in [0,1] so PSNR is directly comparable.',
    modality: 'image',
    task: 'image_map',
    source: 'preset',
    presetKey: 'denoiseimg',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [C, H, W],
    labels: [],
    meta,
    featureSpec: { kind: 'rgb_unit' },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'image_denoise', total, degradation: cfg.degradation, seed: cfg.seed, reference: REFERENCE_DATASETS.denoiseimg },
    config: Object.assign({}, cfg, { peak: 1 })
  });
}

export function genDetection(opts) {
  const cfg = Object.assign({ seed: 188, size: 'medium', imageSize: 48, grid: 6, kinds: ['face', 'hand'] }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 60, medium: 150, large: 360 }[cfg.size] || 150;
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const x = new Float32Array(total * C * H * W);
  const boxes = [];
  const y = new Float32Array(total * 5);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const kind = cfg.kinds[rng.int(cfg.kinds.length)];
    const hasObject = rng.next() < 0.82;
    let res;
    if (!hasObject) {
      const img = new Float32Array(C * H * W);
      const base = rng.range(0.05, 0.6);
      for (let k = 0; k < img.length; k++) img[k] = clamp01(base + rng.range(-0.05, 0.05));
      res = { image: img, box: null };
    } else {
      res = kind === 'face' ? synthFace(rng, H, W, true) : synthHand(rng, H, W, true);
    }
    x.set(res.image, i * C * H * W);
    boxes.push(res.box ? [res.box] : []);
    y.set([res.box ? 1 : 0, res.box ? res.box.x : 0, res.box ? res.box.y : 0, res.box ? res.box.w : 0, res.box ? res.box.h : 0], i * 5);
    meta.push({ preview: hasObject ? kind + ' present' : 'background only', text: '', kind });
  }
  return createDataset({
    name: 'Face and hand detection',
    slug: 'face-hand-detection',
    description: 'Synthetic faces and hands on cluttered backgrounds, with normalised bounding boxes. ' + cfg.grid + '×' + cfg.grid + ' detection grid, five channels per cell: objectness, centre x, centre y, width, height. 18% of images are background-only so false positives are measured.',
    modality: 'image',
    task: 'detection',
    source: 'preset',
    presetKey: 'detection',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [5],
    boxes,
    labels: ['object'],
    meta,
    featureSpec: { kind: 'rgb_unit', grid: cfg.grid },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'detection', total, grid: cfg.grid, seed: cfg.seed, reference: REFERENCE_DATASETS.detection },
    config: Object.assign({}, cfg, { confThreshold: 0.35 })
  });
}

export function genSegmentation(opts) {
  const cfg = Object.assign({ seed: 199, size: 'medium', imageSize: 48 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 60, medium: 150, large: 340 }[cfg.size] || 150;
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const x = new Float32Array(total * C * H * W);
  const y = new Float32Array(total * H * W);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const res = synthSubject(rng, H, W);
    x.set(res.image, i * C * H * W);
    y.set(res.mask, i * H * W);
    let cover = 0;
    for (let k = 0; k < res.mask.length; k++) cover += res.mask[k];
    meta.push({ preview: 'foreground covers ' + Math.round(100 * cover / res.mask.length) + '% of the frame', text: '' });
  }
  return createDataset({
    name: 'Background segmentation',
    slug: 'background-segmentation',
    description: 'Single-subject frames on patterned backgrounds with per-pixel foreground masks. This is the video-call blur case: the mask has to hold up at frame rate, so the model is deliberately small.',
    modality: 'image',
    task: 'mask',
    source: 'preset',
    presetKey: 'segmentation',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [1, H, W],
    labels: ['background', 'foreground'],
    meta,
    featureSpec: { kind: 'rgb_unit' },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'segmentation', total, seed: cfg.seed, reference: REFERENCE_DATASETS.segmentation },
    config: cfg
  });
}

export function genStyleTransfer(opts) {
  const cfg = Object.assign({ seed: 210, size: 'medium', imageSize: 32, styleSeed: 4242 }, opts || {});
  const rng = new Rng(cfg.seed);
  const styleRng = new Rng(cfg.styleSeed);
  const style = { levels: 5, brush: 2, warp: 1.6 };
  const total = { small: 70, medium: 170, large: 380 }[cfg.size] || 170;
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const x = new Float32Array(total * C * H * W);
  const y = new Float32Array(total * C * H * W);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const content = toUnit(synthNatural(rng, C, H, W));
    const styled = stylize(content, C, H, W, styleRng, style);
    x.set(content, i * C * H * W);
    y.set(styled, i * C * H * W);
    meta.push({ preview: 'content → stylised', text: '' });
  }
  return createDataset({
    name: 'Fast style transfer pairs',
    slug: 'style-transfer',
    description: 'Content images paired with a deterministic stylisation: palette quantisation, brush-scale spatial warp and grain, all driven by style seed ' + cfg.styleSeed + '. One forward pass per image, no iterative optimisation, which is exactly what a shipping filter needs.',
    modality: 'image',
    task: 'image_map',
    source: 'preset',
    presetKey: 'style',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [C, H, W],
    labels: [],
    meta,
    featureSpec: { kind: 'rgb_unit' },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'style_transfer', total, styleSeed: cfg.styleSeed, seed: cfg.seed, reference: REFERENCE_DATASETS.style },
    config: Object.assign({}, cfg, { peak: 1 })
  });
}

export function genSaliency(opts) {
  const cfg = Object.assign({ seed: 221, size: 'medium', imageSize: 48 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 60, medium: 150, large: 340 }[cfg.size] || 150;
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const x = new Float32Array(total * C * H * W);
  const y = new Float32Array(total * H * W);
  const y2 = new Float32Array(total * 4);
  const meta = [];
  for (let i = 0; i < total; i++) {
    const res = synthSaliency(rng, H, W);
    x.set(res.image, i * C * H * W);
    y.set(res.saliency, i * H * W);
    const p = res.primary;
    const bw = p.r * 2 / W;
    const bh = p.r * 2 / H;
    y2.set([Math.max(0, p.cx / W - bw / 2), Math.max(0, p.cy / H - bh / 2), bw, bh], i * 4);
    meta.push({ preview: 'primary subject at ' + (100 * p.cx / W).toFixed(0) + '%, ' + (100 * p.cy / H).toFixed(0) + '%', text: '' });
  }
  return createDataset({
    name: 'Saliency and smart crop',
    slug: 'saliency',
    description: 'Images with one or two high-contrast subjects on a textured background. Targets are a continuous saliency map and the normalised crop box of the primary subject, which is what a thumbnail cropper actually consumes.',
    modality: 'image',
    task: 'mask',
    source: 'preset',
    presetKey: 'saliency',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [1, H, W],
    labels: ['background', 'salient'],
    meta,
    featureSpec: { kind: 'rgb_unit' },
    splitMethod: 'random',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'saliency', total, seed: cfg.seed, reference: REFERENCE_DATASETS.saliency },
    config: Object.assign({}, cfg, { cropBoxes: Array.from(y2) })
  });
}

export function genNsfwStandIn(opts) {
  const cfg = Object.assign({ seed: 232, size: 'medium', imageSize: 32 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 90, medium: 220, large: 520 }[cfg.size] || 220;
  const perClass = Math.floor(total / 3);
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const kinds = [2, 1, 0];
  const rows = [];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < perClass; i++) {
      rows.push({ image: toUnit(explicitLikePattern(rng, C, H, W, kinds[c])), label: c });
    }
  }
  const shuffled = rng.shuffle(rows);
  const x = new Float32Array(shuffled.length * C * H * W);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(r.image, i * C * H * W);
    y[i] = r.label;
    meta.push({ preview: ['safe', 'suggestive', 'explicit'][r.label] + ' structural class', text: '', label: r.label });
  });
  return createDataset({
    name: 'Content moderation stand-in',
    slug: 'nsfw-standin',
    description: 'Three structural classes rendered as abstract pattern families. These are stand-ins: they exercise the pipeline, the metrics and the export path end to end. A moderation model you actually ship must be trained on a lawfully licensed, human-reviewed corpus, and needs a documented appeals process alongside it.',
    modality: 'image',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'nsfw',
    license: Object.assign({}, CC0),
    x,
    xShape: [C, H, W],
    y,
    yShape: [1],
    labels: ['safe', 'suggestive', 'explicit'],
    meta,
    featureSpec: { kind: 'rgb_unit' },
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'nsfw_stand_in', perClass, seed: cfg.seed, standIn: true, reference: REFERENCE_DATASETS.nsfw },
    config: cfg
  });
}

export function genImuActivity(opts) {
  const cfg = Object.assign({ seed: 243, size: 'medium', windowLen: 64, sampleRate: 50, channels: 6, activities: null }, opts || {});
  const rng = new Rng(cfg.seed);
  const activities = cfg.activities || IMU_ACTIVITIES;
  const perClass = { small: 30, medium: 72, large: 170 }[cfg.size] || 72;
  const rows = [];
  for (const act of activities) {
    for (let i = 0; i < perClass; i++) {
      const w = imuWindow(rng, act, cfg.windowLen, cfg.sampleRate, cfg.channels);
      const jitter = rng.range(0.92, 1.08);
      for (let k = 0; k < w.length; k++) w[k] = w[k] * jitter + rng.gauss() * 0.012;
      rows.push({ w, act });
    }
  }
  const shuffled = rng.shuffle(rows);
  const per = cfg.windowLen * cfg.channels;
  const x = new Float32Array(shuffled.length * per);
  const y = new Float32Array(shuffled.length);
  const meta = [];
  shuffled.forEach((r, i) => {
    x.set(r.w, i * per);
    y[i] = activities.indexOf(r.act);
    meta.push({ preview: r.act, text: r.act });
  });
  return createDataset({
    name: 'Accelerometer activity windows',
    slug: 'imu-activity',
    description: activities.length + ' activities at ' + cfg.sampleRate + ' Hz over ' + (cfg.windowLen / cfg.sampleRate).toFixed(2) + 's windows, six channels. Generated from a biomechanical model per activity, so the difference between walking and running is cadence and amplitude, not a label shortcut.',
    modality: 'sensor',
    task: 'multiclass',
    source: 'preset',
    presetKey: 'imu',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.windowLen, cfg.channels],
    y,
    yShape: [1],
    labels: activities,
    meta,
    featureSpec: { kind: 'imu_window', sampleRate: cfg.sampleRate, channels: cfg.channels },
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    sampleRate: cfg.sampleRate,
    units: ['g', 'g', 'g', 'rad/s', 'rad/s', 'rad/s'],
    provenance: { generator: 'imu', activities, perClass, seed: cfg.seed, reference: REFERENCE_DATASETS.imu },
    config: cfg
  });
}

export function genAnomalyTelemetry(opts) {
  const cfg = Object.assign({ seed: 254, size: 'medium', windowLen: 32, dims: 6, trainContamination: 0.02 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 220, medium: 520, large: 1200 }[cfg.size] || 520;
  const kinds = ['spike', 'level_shift', 'oscillation', 'flatline', 'correlation_break'];
  const per = cfg.windowLen * cfg.dims;
  const x = new Float32Array(total * per);
  const y = new Float32Array(total);
  const meta = [];
  const anomalyCount = Math.round(total * 0.22);
  for (let i = 0; i < total; i++) {
    const isAnomaly = i < anomalyCount;
    const kind = isAnomaly ? kinds[rng.int(kinds.length)] : 'normal';
    const res = anomalyWindow(rng, cfg.dims, cfg.windowLen, kind);
    x.set(res.data, i * per);
    y[i] = isAnomaly ? 1 : 0;
    meta.push({ preview: isAnomaly ? kind : 'nominal', text: '', kind });
  }
  const order = rng.shuffle(Array.from({ length: total }, (_, i) => i));
  const xShuffled = new Float32Array(total * per);
  const yShuffled = new Float32Array(total);
  const metaShuffled = [];
  order.forEach((src, i) => {
    xShuffled.set(x.subarray(src * per, (src + 1) * per), i * per);
    yShuffled[i] = y[src];
    metaShuffled.push(meta[src]);
  });
  const ds = createDataset({
    name: 'Service telemetry anomalies',
    slug: 'anomaly-telemetry',
    description: cfg.dims + '-channel service metrics over ' + cfg.windowLen + '-step windows. The autoencoder learns the nominal manifold and reconstruction error becomes the anomaly score. Five failure modes are injected: spikes, level shifts, oscillation, flatline and broken cross-channel correlation.',
    modality: 'series',
    task: 'anomaly',
    source: 'preset',
    presetKey: 'anomaly',
    license: Object.assign({}, CC0),
    x: xShuffled,
    xShape: [cfg.windowLen, cfg.dims],
    y: yShuffled,
    yShape: [1],
    labels: ['nominal', 'anomalous'],
    meta: metaShuffled,
    featureSpec: { kind: 'metric_window', dims: cfg.dims, columns: ['cpu', 'memory', 'disk_io', 'net_in', 'net_out', 'latency'].slice(0, cfg.dims) },
    splitMethod: 'random',
    splitRatios: [0.66, 0.17, 0.17],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'anomaly', total, anomalyCount, seed: cfg.seed, reference: REFERENCE_DATASETS.anomaly },
    config: cfg
  });
  const train = [];
  const held = [];
  for (const i of ds.splits.train) (yShuffled[i] === 0 || rng.next() < cfg.trainContamination ? train : held).push(i);
  const val = Array.from(ds.splits.val);
  const test = Array.from(ds.splits.test).concat(held);
  ds.splits = { train: Int32Array.from(train), val: Int32Array.from(val), test: Int32Array.from(test) };
  ds.splitMethod = 'anomaly-clean-train';
  recompute(ds);
  ds.provenance.trainNote = 'Training split is nominal-only apart from ' + Math.round(cfg.trainContamination * 100) + '% contamination, so the reconstruction baseline is not polluted. Validation and test carry the full anomaly mix.';
  return ds;
}

export function genForecasting(opts) {
  const cfg = Object.assign({ seed: 265, size: 'medium', history: 48, horizon: 8, kinds: ['traffic', 'battery', 'sales', 'energy'] }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 160, medium: 380, large: 900 }[cfg.size] || 380;
  const per = cfg.history;
  const x = new Float32Array(total * per);
  const y = new Float32Array(total * cfg.horizon);
  const meta = [];
  const kindsUsed = [];
  for (let i = 0; i < total; i++) {
    const kind = cfg.kinds[rng.int(cfg.kinds.length)];
    const series = forecastSeries(rng, kind, cfg.history, cfg.horizon);
    x.set(series.subarray(0, cfg.history), i * per);
    y.set(series.subarray(cfg.history, cfg.history + cfg.horizon), i * cfg.horizon);
    kindsUsed.push(kind);
    meta.push({ preview: kind + ' series · ' + cfg.history + ' in, ' + cfg.horizon + ' out', text: '', kind });
  }
  return createDataset({
    name: 'Short-horizon forecasting',
    slug: 'forecasting',
    description: 'Four series families — request traffic, battery drain, retail sales and energy load — each with trend, seasonality, occasional spikes and observation noise. The model sees ' + cfg.history + ' steps and predicts the next ' + cfg.horizon + '.',
    modality: 'series',
    task: 'forecast',
    source: 'preset',
    presetKey: 'forecast',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.history, 1],
    y,
    yShape: [cfg.horizon],
    labels: [],
    meta,
    featureSpec: { kind: 'series_window', history: cfg.history, horizon: cfg.horizon, families: cfg.kinds },
    splitMethod: 'temporal',
    splitRatios: [0.72, 0.14, 0.14],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'forecasting', total, seed: cfg.seed, reference: REFERENCE_DATASETS.forecast },
    config: Object.assign({}, cfg, { kindsUsed })
  });
}

export function genBotDetection(opts) {
  const cfg = Object.assign({ seed: 276, size: 'medium', points: 48 }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 180, medium: 420, large: 980 }[cfg.size] || 420;
  const kinds = [
    { key: 'human', weight: 0.46, label: 0 },
    { key: 'bezier_bot', weight: 0.2, label: 1 },
    { key: 'linear_bot', weight: 0.18, label: 1 },
    { key: 'macro', weight: 0.09, label: 1 },
    { key: 'hammer', weight: 0.07, label: 1 }
  ];
  const rows = [];
  let cum = 0;
  const cumWeights = kinds.map((k) => (cum += k.weight));
  for (let i = 0; i < total; i++) {
    const r = rng.next();
    let ki = 0;
    while (ki < cumWeights.length - 1 && r > cumWeights[ki]) ki++;
    const kind = kinds[ki];
    const mouse = mouseTrace(rng, kind.key === 'macro' || kind.key === 'hammer' ? 'human' : kind.key, cfg.points);
    const keys = keystrokeTrace(rng, kind.key === 'macro' ? 'macro' : kind.key === 'hammer' ? 'hammer' : kind.key === 'human' ? 'human' : 'linear_bot', cfg.points);
    const mouseFeatures = behaviorFeatures(mouse, cfg.points);
    const keyFeatures = behaviorFeatures(keys, cfg.points);
    const combined = new Float32Array(mouseFeatures.length + keyFeatures.length);
    combined.set(mouseFeatures);
    combined.set(keyFeatures, mouseFeatures.length);
    let label = kind.label;
    if (rng.next() < 0.1) {
      for (let k = 0; k < combined.length; k++) combined[k] += rng.gauss() * 0.9;
      label = rng.next() < 0.5 ? label : 1 - label;
    }
    rows.push({ features: combined, kind: kind.key, label });
  }
  const dim = rows[0].features.length;
  const x = new Float32Array(rows.length * dim);
  const y = new Float32Array(rows.length);
  const meta = [];
  rows.forEach((r, i) => {
    x.set(r.features, i * dim);
    y[i] = r.label;
    meta.push({ preview: r.kind + (r.label ? ' (automated)' : ' (human)'), text: r.kind, kind: r.kind });
  });
  const ds = createDataset({
    name: 'Behavioural bot detection',
    slug: 'bot-detection',
    description: 'Pointer traces and keystroke timing reduced to summary statistics per channel: mean, spread, extremes, deciles, skew, kurtosis, zero-crossing rate, lag-1 autocorrelation and histogram entropy. Five actor types, one binary decision. Ten percent of samples get heavy jitter and an ambiguous label so the achievable ceiling stays below perfect.',
    modality: 'tabular',
    task: 'binary',
    source: 'preset',
    presetKey: 'bot',
    license: Object.assign({}, CC0),
    x,
    xShape: [dim],
    y,
    yShape: [1],
    labels: ['human', 'automated'],
    meta,
    featureSpec: { kind: 'behaviour_stats', points: cfg.points, dim },
    splitMethod: 'stratified',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'bot', total, kinds: kinds.map((k) => k.key), seed: cfg.seed, reference: REFERENCE_DATASETS.bot },
    config: cfg
  });
  return standardizeDataset(ds);
}

export function genByteCompression(opts) {
  const cfg = Object.assign({ seed: 287, size: 'medium', seqLen: 96, mix: 'mixed' }, opts || {});
  const rng = new Rng(cfg.seed);
  const total = { small: 220, medium: 620, large: 1500 }[cfg.size] || 620;
  const sources = [];
  if (cfg.mix !== 'code') sources.push(LM_CORPUS.join(' \n '));
  if (cfg.mix !== 'text') sources.push(Object.values(CODE_SNIPPETS).flat().join('\n'));
  const corpus = sources.join('\n');
  const bytes = new TextEncoder().encode(corpus);
  const x = new Float32Array(total * cfg.seqLen);
  const y = new Float32Array(total);
  const lengths = new Int32Array(total);
  const meta = [];
  const counts = new Array(256).fill(0);
  for (let b of bytes) counts[b]++;
  const starts = collectUnique(total, () => {
    const start = rng.int(Math.max(1, bytes.length - cfg.seqLen));
    return bytes.length - start >= cfg.seqLen - 4 ? start : null;
  }, (v) => v, total * 200);
  for (let i = 0; i < starts.length; i++) {
    const start = starts[i];
    const len = Math.min(cfg.seqLen, bytes.length - start);
    for (let k = 0; k < len; k++) x[i * cfg.seqLen + k] = bytes[start + k] + 3;
    lengths[i] = len;
    y[i] = len;
    meta.push({ preview: new TextDecoder().decode(bytes.subarray(start, start + Math.min(len, 60))).replace(/\n/g, ' '), text: '' });
  }
  let h0 = 0;
  for (const c of counts) {
    if (!c) continue;
    const p = c / bytes.length;
    h0 -= p * Math.log2(p);
  }
  const vocabSize = 259;
  return createDataset({
    name: 'Byte stream compression corpus',
    slug: 'byte-compression',
    task: 'lm',
    description: 'Raw byte sequences for a next-byte model feeding an arithmetic coder. Baseline order-0 entropy of the corpus is ' + h0.toFixed(3) + ' bits per byte, which is the number the trained model has to beat.',
    modality: 'bytes',
    source: 'preset',
    presetKey: 'compression',
    license: Object.assign({}, CC0),
    x,
    xShape: [cfg.seqLen],
    y,
    yShape: [1],
    lengths,
    labels: [],
    meta,
    featureSpec: { kind: 'byte_ids', vocab: vocabSize, maxLen: cfg.seqLen, offset: 3 },
    splitMethod: 'random',
    splitRatios: [0.8, 0.1, 0.1],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'compression', total, corpusBytes: bytes.length, baselineBitsPerByte: h0, seed: cfg.seed, reference: REFERENCE_DATASETS.compression },
    config: Object.assign({}, cfg, { baselineBitsPerByte: h0, vocabSize })
  });
}

export function genNeuralTexture(opts) {
  const cfg = Object.assign({ seed: 298, imageSize: 48, samples: 4000 }, opts || {});
  const rng = new Rng(cfg.seed);
  const imgRng = new Rng(cfg.seed + 7);
  const C = 3;
  const H = cfg.imageSize;
  const W = cfg.imageSize;
  const image = toUnit(synthNatural(imgRng, C, H, W));
  const total = Math.min(cfg.samples, H * W);
  const positions = rng.shuffle(Array.from({ length: H * W }, (_, i) => i)).slice(0, total);
  const FREQS = [1, 2, 3, 5, 8, 13, 21];
  const dim = 3 + FREQS.length * 4;
  const x = new Float32Array(total * dim);
  const y = new Float32Array(total * C);
  positions.forEach((p, i) => {
    const py = Math.floor(p / W);
    const px = p % W;
    const u = (px / (W - 1)) * 2 - 1;
    const v = (py / (H - 1)) * 2 - 1;
    x[i * dim] = u;
    x[i * dim + 1] = v;
    x[i * dim + 2] = Math.sqrt(u * u + v * v);
    for (let k = 0; k < FREQS.length; k++) {
      const f = FREQS[k] * Math.PI;
      x[i * dim + 3 + k * 4] = Math.sin(f * u);
      x[i * dim + 4 + k * 4] = Math.cos(f * u);
      x[i * dim + 5 + k * 4] = Math.sin(f * v);
      x[i * dim + 6 + k * 4] = Math.cos(f * v);
    }
    for (let c = 0; c < C; c++) y[i * C + c] = image[(c * H + py) * W + px];
  });
  return createDataset({
    name: 'Neural texture coordinates',
    slug: 'neural-texture',
    task: 'coordinate_field',
    description: 'Every sample is one pixel of a single ' + H + '×' + W + ' image. The input is the normalised coordinate plus ' + FREQS.length + ' octaves of sine and cosine positional encoding, and the output is RGB. A trained network is the image, so the interesting comparison is bytes of weights against bytes of pixels.',
    modality: 'image',
    source: 'preset',
    presetKey: 'texture',
    license: Object.assign({}, CC0),
    x,
    xShape: [dim],
    y,
    yShape: [C],
    labels: [],
    meta: null,
    featureSpec: { kind: 'coordinate_samples', imageSize: [H, W], channels: C, dim, freqs: FREQS },
    splitMethod: 'random',
    splitRatios: [0.7, 0.15, 0.15],
    splitSeed: cfg.seed + 1,
    provenance: { generator: 'texture', total, imageSize: [H, W], positionalEncoding: FREQS, seed: cfg.seed, reference: REFERENCE_DATASETS.texture },
    config: Object.assign({}, cfg, { image: Array.from(image), imageSize: [H, W], channels: C, peak: 1 })
  });
}

const SIZE_FIELD = { key: 'size', label: 'Size', type: 'select', def: 'medium', options: ['small', 'medium', 'large'] };
const SEED_FIELD = { key: 'seed', label: 'Seed', type: 'int', def: 1, min: 1, max: 999999 };
const num = (key, label, def, min, max, step) => ({ key, label, type: step && step < 1 ? 'float' : 'int', def, min, max, step: step || 1 });
const sel = (key, label, def, options) => ({ key, label, type: 'select', def, options });

export const GENERATORS = {
  langid: {
    key: 'langid', label: 'Language identification', modality: 'text', task: 'multiclass', refKey: 'langid', fn: genLangId,
    headline: '12 languages · hashed char n-grams',
    blurb: 'Sentences in twelve languages, expanded with a word-level Markov model fitted per language so the generated text keeps real word shapes rather than character soup. Features are hashed character n-grams, which means no vocabulary file has to ship with the model.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 640, 128, 2048, 64)]
  },
  intent: {
    key: 'intent', label: 'Intent classification', modality: 'text', task: 'multiclass', refKey: 'intent', fn: genIntent,
    headline: '18 intents · slot-rich templates',
    blurb: 'Customer-assistant utterances built from intent templates and slot values, then perturbed with typos, dropped words and synonym swaps so the classifier has to read meaning rather than match strings.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 900, 256, 2048, 64), num('noise', 'Perturbation rate', 0.08, 0, 0.4, 0.01)]
  },
  spam: {
    key: 'spam', label: 'Spam and toxicity', modality: 'text', task: 'multiclass', refKey: 'spam', fn: genSpamToxicity,
    headline: '3 classes · optional label noise',
    blurb: 'Ham, spam and toxic messages with a configurable label-noise rate, because a moderation classifier that has never seen a wrong label will look better in the browser than it does in production.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 768, 256, 2048, 64), sel('classes', 'Classes', 3, [2, 3, 4]), num('labelNoise', 'Label noise', 0.02, 0, 0.25, 0.01)]
  },
  pii: {
    key: 'pii', label: 'PII redaction', modality: 'text', task: 'seq_tag', refKey: 'pii', fn: genPii,
    headline: '7 entity types · BIO character tags',
    blurb: 'Sentences with emails, phone numbers, card numbers, API keys, names, postal addresses and dates inserted into them. Labels are BIO tags per character, which is the granularity a redaction filter actually needs before text reaches a log.',
    metrics: ['entity_f1', 'token_accuracy', 'macro_f1'],
    options: [SIZE_FIELD, SEED_FIELD, num('maxLen', 'Max characters', 96, 48, 256, 8), num('piiRate', 'Second entity rate', 0.62, 0, 1, 0.01)]
  },
  code: {
    key: 'code', label: 'Code language', modality: 'text', task: 'multiclass', refKey: 'code', fn: genCodeLanguage,
    headline: '16 languages · identifier mutation',
    blurb: 'Short source snippets in sixteen languages. Identifiers are renamed through a synonym table and fragments are recombined, so each generated row is unique and the model cannot win by memorising variable names.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 1024, 256, 4096, 128)]
  },
  password: {
    key: 'password', label: 'Password strength', modality: 'text', task: 'multiclass', refKey: 'password', fn: genPasswordStrength,
    headline: '5 bands · guess-number features',
    blurb: 'Passwords scored into five strength bands from character-class features, dictionary hits, keyboard walks and an estimated guess rank. The reference corpora for this task are leaked password dumps and are not usable commercially, so everything here is synthesised.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 320, 96, 1024, 32)]
  },
  embeddings: {
    key: 'embeddings', label: 'Sentence embeddings', modality: 'text', task: 'embedding', refKey: 'embeddings', fn: genSentenceEmbeddings,
    headline: '40 meaning groups · contrastive pairs',
    blurb: 'Paraphrase pairs grouped by meaning, each group expanded into surface variations. Trained with a group-aware contrastive loss so two rewordings of the same request land next to each other and the group ids keep the split honest.',
    metrics: ['pair_auc', 'paraphrase_recall_at_1', 'recall_at_1'],
    options: [SIZE_FIELD, SEED_FIELD, num('dim', 'Feature dimensions', 512, 128, 2048, 64)]
  },
  autocomplete: {
    key: 'autocomplete', label: 'Character autocomplete', modality: 'text', task: 'lm', refKey: 'lm', fn: genAutocomplete,
    headline: 'Next character · 128 symbol vocab',
    blurb: 'Character sequences for a next-symbol model, which is the smallest thing that can power inline autocomplete. Reports perplexity and bits per symbol alongside the compression those numbers imply.',
    metrics: ['perplexity', 'bits_per_symbol', 'top1_accuracy'],
    options: [SIZE_FIELD, SEED_FIELD, num('seqLen', 'Sequence length', 64, 16, 128, 8)]
  },
  wake: {
    key: 'wake', label: 'Wake word', modality: 'audio', task: 'binary', refKey: 'wake', fn: genWakeWord,
    headline: '2 keywords · 5 negative phrases',
    blurb: 'Keyword spotting clips at a fixed length so every row has the same frame grid. Negatives are other phrases and coloured noise, and the reported numbers are the ones a wake-word team cares about: false accepts and false rejects per hour of audio.',
    metrics: ['auc_roc', 'f1', 'far', 'average_precision'],
    options: [SIZE_FIELD, SEED_FIELD, num('snrMin', 'Min SNR dB', 3, -10, 30, 1), num('snrMax', 'Max SNR dB', 22, 0, 45, 1)]
  },
  vad: {
    key: 'vad', label: 'Voice activity', modality: 'audio', task: 'frame_binary', refKey: 'vad', fn: genVad,
    headline: 'Per-frame speech / non-speech',
    blurb: 'Per-frame speech detection over log-mel features, the Silero-shaped problem: one label per 20 ms frame rather than one per clip, scored with frame AUC plus the false-accept and false-reject rates at the chosen threshold.',
    metrics: ['auc_roc', 'frame_f1', 'far', 'frr'],
    options: [SIZE_FIELD, SEED_FIELD]
  },
  denoise: {
    key: 'denoise', label: 'Noise suppression', modality: 'audio', task: 'spectral_mask', refKey: 'denoise', fn: genNoiseSuppression,
    headline: 'Clean / noisy spectral pairs',
    blurb: 'Clean and noisy speech pairs where the target is the ideal magnitude mask. Reconstruction runs through an inverse STFT with the noisy phase, so SI-SDR, segmental SNR and the perceptual estimate are measured on real waveforms, not on spectra.',
    metrics: ['si_sdr_improvement', 'si_sdr', 'pesq_approx', 'mask_mae'],
    options: [SIZE_FIELD, SEED_FIELD, num('snrMin', 'Min SNR dB', -3, -15, 20, 1), num('snrMax', 'Max SNR dB', 12, 0, 35, 1)]
  },
  pitch: {
    key: 'pitch', label: 'Pitch class', modality: 'audio', task: 'pitch_class', refKey: 'pitch', fn: genPitch,
    headline: '36 chromatic classes · 60-1400 Hz',
    blurb: 'Synthesised tones with one to six harmonics, vibrato and additive noise. Features are a 2048-point log magnitude spectrum restricted to the 60-1400 Hz band so harmonics stay resolvable, and the headline metric is cent error rather than plain accuracy.',
    metrics: ['raw_pitch_accuracy', 'mean_abs_cent_error', 'octave_error_rate', 'accuracy'],
    options: [SIZE_FIELD, SEED_FIELD, num('harmonics', 'Max harmonics', 6, 1, 12, 1)]
  },
  events: {
    key: 'events', label: 'Sound events', modality: 'audio', task: 'multiclass', refKey: 'events', fn: genSoundEvents,
    headline: '12 events · doorbell to glass break',
    blurb: 'Twelve everyday sound events synthesised from spectral recipes, mixed with coloured noise at random signal-to-noise ratios and cut to a fixed frame grid so the batcher never has to pad.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('snrMin', 'Min SNR dB', 4, -10, 30, 1), num('snrMax', 'Max SNR dB', 26, 0, 45, 1)]
  },
  onset: {
    key: 'onset', label: 'Beat and onset', modality: 'audio', task: 'onset', refKey: 'onset', fn: genBeatOnset,
    headline: 'Rhythm tracks · per-frame onsets',
    blurb: 'Rhythm tracks at 70 to 160 BPM with per-frame onset labels, scored with an onset F1 that allows a small timing tolerance, plus a tempo estimate in BPM. Exact-frame matching would punish a model for being 10 ms early.',
    metrics: ['onset_f1', 'auc_roc', 'tempo_mae_bpm', 'frame_f1'],
    options: [SIZE_FIELD, SEED_FIELD]
  }
};

Object.assign(GENERATORS, {
  superres: {
    key: 'superres', label: 'Super-resolution', modality: 'image', task: 'image_map', refKey: 'superres', fn: genSuperResolution,
    headline: 'Patches · residual upsampler',
    blurb: 'Procedural images carrying sharp polygons, lines and grain, downsampled by area averaging, brought back with bicubic interpolation and cut into non-overlapping patches. The bicubic result is both the input and the baseline, so a positive PSNR gain means the network recovered edge detail that interpolation cannot.',
    metrics: ['psnr_gain', 'psnr', 'ssim', 'baseline_psnr'],
    options: [SIZE_FIELD, SEED_FIELD, sel('factor', 'Upscale factor', 2, [2, 3, 4]), num('hrSize', 'Source size', 32, 16, 64, 8), num('patchSize', 'Patch size', 16, 8, 32, 4)]
  },
  denoiseimg: {
    key: 'denoiseimg', label: 'Image denoise / JPEG cleanup', modality: 'image', task: 'image_map', refKey: 'denoiseimg', fn: genImageDenoise,
    headline: 'Gaussian, Poisson, JPEG, mixed',
    blurb: 'Clean and degraded image pairs with selectable degradation: Gaussian noise, Poisson-like noise, coarse quantisation standing in for JPEG blocking, or a mixture of all three. Scored against the degraded input so the gain is visible.',
    metrics: ['psnr_gain', 'psnr', 'ssim', 'baseline_psnr'],
    options: [SIZE_FIELD, SEED_FIELD, sel('degradation', 'Degradation', 'mixed', ['gaussian', 'poisson', 'jpeg', 'mixed']), num('noiseStd', 'Noise level', 0.16, 0.02, 0.4, 0.01)]
  },
  detection: {
    key: 'detection', label: 'Face and hand detection', modality: 'image', task: 'detection', refKey: 'detection', fn: genDetection,
    headline: 'BlazeFace-shaped grid output',
    blurb: 'Synthetic faces and hands on noisy backgrounds with tight boxes, plus background-only negatives. The grid output is objectness in channel zero and box coordinates in the rest, scored with mAP at 0.5 IoU and a confidence sweep for the operating point.',
    metrics: ['mAP50', 'mAP', 'recall', 'precision', 'f1_at_best_conf'],
    options: [SIZE_FIELD, SEED_FIELD, num('imageSize', 'Image size', 48, 24, 96, 8), num('grid', 'Detection grid', 6, 3, 12, 1)]
  },
  segmentation: {
    key: 'segmentation', label: 'Background segmentation', modality: 'image', task: 'mask', refKey: 'segmentation', fn: genSegmentation,
    headline: 'Subject / background masks',
    blurb: 'A subject on a cluttered background with a pixel-exact mask, scored with IoU, Dice and a boundary AUC that separates "found the object" from "found its edges". Foreground is a minority of pixels, so the loss needs class weighting.',
    metrics: ['iou', 'dice', 'pixel_accuracy', 'boundary_auc'],
    options: [SIZE_FIELD, SEED_FIELD, num('imageSize', 'Image size', 48, 24, 96, 8)]
  },
  style: {
    key: 'style', label: 'Single-pass style transfer', modality: 'image', task: 'image_map', refKey: 'style', fn: genStyleTransfer,
    headline: 'One fixed target style',
    blurb: 'Content images paired with one procedurally stylised version of themselves, which is what makes a single feed-forward pass possible instead of per-image optimisation. The baseline is the unstyled content image.',
    metrics: ['psnr_gain', 'psnr', 'ssim'],
    options: [SIZE_FIELD, SEED_FIELD, num('imageSize', 'Image size', 32, 16, 64, 8), num('styleSeed', 'Style seed', 4242, 1, 99999, 1)]
  },
  saliency: {
    key: 'saliency', label: 'Smart crop / saliency', modality: 'image', task: 'mask', refKey: 'saliency', fn: genSaliency,
    headline: 'Heatmap for crop boxes',
    blurb: 'Images with a salient subject and a soft importance heatmap. The heatmap is what a smart-crop pipeline integrates to choose a crop box, so IoU on the thresholded map and boundary AUC both matter.',
    metrics: ['iou', 'dice', 'boundary_auc', 'pixel_accuracy'],
    options: [SIZE_FIELD, SEED_FIELD, num('imageSize', 'Image size', 48, 24, 96, 8)]
  },
  nsfw: {
    key: 'nsfw', label: 'NSFW filter stand-in', modality: 'image', task: 'multiclass', refKey: 'nsfw', fn: genNsfwStandIn,
    headline: '3 classes · synthetic stand-in',
    blurb: 'A structural stand-in for an NSFW classifier: three classes of synthetic imagery that differ in skin-tone ratio, edge density and region layout. It exercises the same pipeline and metrics as a real filter without shipping or generating explicit content.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('imageSize', 'Image size', 32, 16, 64, 8)]
  },
  imu: {
    key: 'imu', label: 'Accelerometer activity', modality: 'sensor', task: 'multiclass', refKey: 'imu', fn: genImuActivity,
    headline: '8 activities · 6-axis windows',
    blurb: 'Six-axis accelerometer and gyroscope windows for eight activities including walking, running, stair climbing, a wrist flick and a fall. Windows are fixed length so a strided convolution over time is all the architecture needs.',
    metrics: ['accuracy', 'macro_f1', 'log_loss'],
    options: [SIZE_FIELD, SEED_FIELD, num('windowLen', 'Window samples', 64, 24, 192, 8), num('sampleRate', 'Sample rate Hz', 50, 20, 100, 5), sel('channels', 'Channels', 6, [3, 6, 9])]
  },
  anomaly: {
    key: 'anomaly', label: 'Anomaly autoencoder', modality: 'series', task: 'anomaly', refKey: 'anomaly', fn: genAnomalyTelemetry,
    headline: 'Reconstruction error scoring',
    blurb: 'Multivariate telemetry windows where normal behaviour is learned by reconstruction and anomalies are scored by error. Reports AUROC, precision at k and recall at a 5 percent false-positive rate, which is how an on-call threshold is actually chosen.',
    metrics: ['auroc', 'average_precision', 'recall_at_5fpr', 'p_at_10'],
    options: [SIZE_FIELD, SEED_FIELD, num('windowLen', 'Window length', 32, 8, 128, 4), num('dims', 'Channels', 6, 2, 16, 1), num('trainContamination', 'Train contamination', 0.02, 0, 0.2, 0.01)]
  },
  forecast: {
    key: 'forecast', label: 'Tiny forecaster', modality: 'series', task: 'forecast', refKey: 'forecast', fn: genForecasting,
    headline: 'Traffic, battery, sales, energy',
    blurb: 'Four series flavours with trend, seasonality and noise, mapped from a history window to a fixed horizon. A naive last-value baseline is reported next to the model so it is obvious whether the network earned its parameters.',
    metrics: ['mae', 'rmse', 'r2', 'naive_mae', 'wape'],
    options: [SIZE_FIELD, SEED_FIELD, num('history', 'History length', 48, 12, 192, 4), num('horizon', 'Horizon', 8, 1, 32, 1)]
  },
  bot: {
    key: 'bot', label: 'Bot detection', modality: 'sensor', task: 'binary', refKey: 'bot', fn: genBotDetection,
    headline: 'Pointer and keystroke statistics',
    blurb: 'Pointer traces and keystroke timing reduced to summary statistics per channel. Five actor types collapse to one binary decision, and a tenth of the samples are deliberately jittered and relabelled so the achievable ceiling stays below perfect.',
    metrics: ['auc_roc', 'average_precision', 'f1', 'far'],
    options: [SIZE_FIELD, SEED_FIELD, num('points', 'Trace points', 48, 16, 160, 8)]
  },
  compression: {
    key: 'compression', label: 'Learned byte compression', modality: 'text', task: 'lm', refKey: 'compression', fn: genByteCompression,
    headline: 'Next byte · arithmetic-coder ready',
    blurb: 'Byte sequences for a next-byte model, the core of a learned compressor: the predicted distribution drives an arithmetic coder, so bits per symbol converts directly into compressed size. An order-0 entropy baseline is reported alongside.',
    metrics: ['bits_per_symbol', 'bytes_per_symbol', 'perplexity', 'top1_accuracy'],
    options: [SIZE_FIELD, SEED_FIELD, num('seqLen', 'Sequence length', 96, 32, 256, 8), sel('mix', 'Corpus mix', 'mixed', ['mixed', 'text', 'code', 'json'])]
  },
  texture: {
    key: 'texture', label: 'Neural texture', modality: 'image', task: 'coordinate_field', refKey: 'texture', fn: genNeuralTexture,
    headline: 'The network is the image',
    blurb: 'Every sample is one pixel of a single image: a coordinate with sine and cosine positional encoding in, RGB out. Once trained the weights are the image, so the comparison that matters is bytes of parameters against bytes of raw pixels.',
    metrics: ['psnr_render', 'psnr', 'ssim', 'compression_ratio', 'r2'],
    options: [SEED_FIELD, num('imageSize', 'Image size', 48, 16, 96, 8), num('samples', 'Pixel samples', 4000, 256, 16384, 64)]
  }
});

Object.assign(GENERATORS, DOMAIN_GENERATORS);
Object.assign(REFERENCE_DATASETS, DOMAIN_REFERENCES);

export const GENERATOR_LIST = Object.keys(GENERATORS).map((k) => GENERATORS[k]);

export function generatorByKey(key) {
  return GENERATORS[key] || null;
}

export function buildDataset(key, opts) {
  const g = GENERATORS[key];
  if (!g) throw new Error('unknown dataset preset: ' + key);
  const ds = g.fn(opts || {});
  ds.presetKey = key;
  return ds;
}

export function generatorsForModality(modality) {
  return GENERATOR_LIST.filter((g) => !modality || modality === 'all' || g.modality === modality);
}

export function referenceFor(key) {
  const g = GENERATORS[key];
  return g ? REFERENCE_DATASETS[g.refKey] || null : null;
}
