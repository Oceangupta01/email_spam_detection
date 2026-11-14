"""
Email spam augmentation data.

These representative examples cover modern email spam categories
that are absent from the original 2005 SMS Spam Collection dataset.
Used by train.py to augment the training corpus.
"""

EMAIL_SPAM_AUGMENTATION = [
    # ---- 419 / Advance Fee Fraud ----
    ("spam", "Dear Friend, I am Barrister Johnson Adeyemi, a lawyer representing a deceased client. He left behind $15.5 million USD in an offshore account. I need your assistance to transfer this fund. Reply for more details."),
    ("spam", "Dear Sir/Madam, I am Prince Emmanuel Okafor from Nigeria. I have $12 million USD that I wish to transfer out of the country. I need a foreign partner. You will receive 30% as your commission."),
    ("spam", "URGENT: My name is Dr. Ahmed Al-Rashid. I am a diplomat with a consignment worth $7.5 million for delivery. Please contact me immediately with your bank details to facilitate release."),
    ("spam", "Confidential business proposal: I have inherited $4.8 million from my late father. Due to legal restrictions, I need a foreign account to transfer this inheritance. You will get 40% of the total sum."),
    ("spam", "Hello, I am a banker and I discovered an unclaimed sum of $9.2 million belonging to a deceased foreigner. I need your cooperation to claim these funds. This is 100% risk-free and legal."),

    # ---- Phishing / Account Takeover ----
    ("spam", "URGENT: Your PayPal account has been temporarily suspended due to unusual activity. Click the link below to verify your identity and restore full access immediately: http://paypal-account-verify.com/secure"),
    ("spam", "Security Alert from Apple: Your Apple ID has been locked due to multiple failed sign-in attempts from Russia. Verify your account now to restore access: https://apple-id-verify-login.info/reset"),
    ("spam", "Your Amazon account has been compromised! Unauthorized purchase detected for $499.99. If you did not authorize this, verify your account immediately at: http://amazon-secure-login.net/verify"),
    ("spam", "Microsoft Security Team: We have detected a new login to your Microsoft account from an unknown device in China. Click here to confirm or deny this login: http://microsoftsupport-login.com"),
    ("spam", "Important notice from your bank: Your account has been temporarily restricted. You must verify your personal information within 24 hours to avoid permanent account closure. Click here now."),
    ("spam", "Your Netflix account payment failed. Update your billing information immediately to continue your subscription: http://netflix-billing-update.com/account"),

    # ---- Pharmaceutical Spam ----
    ("spam", "SAVE 80% on prescription medications! Generic Viagra, Cialis, Levitra available without prescription. Fast discreet shipping worldwide. Order now at http://cheapmeds-online.com"),
    ("spam", "Online Pharmacy - No Prescription Required! Buy Viagra 100mg, Cialis 20mg at lowest prices. FDA approved generic medications. Discreet packaging. Free shipping on orders over $50."),
    ("spam", "Get the medications you need without expensive doctor visits! Sildenafil, Tadalafil, weight loss pills, sleeping aids - all available online with no prescription required."),
    ("spam", "Special offer: Buy 2 get 1 free on all erectile dysfunction medications! Genuine generic pills at 70% off retail price. No prescription needed. Visit our secure pharmacy today."),

    # ---- Cryptocurrency & Investment Scams ----
    ("spam", "BITCOIN INVESTMENT OPPORTUNITY: Turn $500 into $5,000 in just 7 days! Our AI trading bot has 97.3% success rate. Limited spots available. Sign up now at crypto-doubler.io"),
    ("spam", "Your cryptocurrency wallet has been flagged for suspicious transactions. Enter your 12-word seed phrase immediately to verify ownership and prevent account lockout: http://wallet-verify.net"),
    ("spam", "Exclusive crypto airdrop! Claim your FREE 2.5 ETH now! Just connect your MetaMask wallet and verify your holdings. Limited time offer - only 500 slots available!"),
    ("spam", "URGENT: Elon Musk's crypto giveaway is LIVE! Send 1 BTC and receive 2 BTC back! This offer expires in 60 minutes. Send to wallet: 1A2b3C4d5E6f7G8h"),
    ("spam", "Guaranteed passive income with crypto! Earn 10-20% daily returns with our automated DeFi trading system. Minimum investment $200. Join 50,000+ satisfied investors today!"),

    # ---- Employment / Work-From-Home Scams ----
    ("spam", "WORK FROM HOME OPPORTUNITY! Earn $500-$1,000 per day doing simple data entry tasks. No experience required. Work on your own schedule. Apply now - limited positions available!"),
    ("spam", "Job Alert: We are hiring remote workers to earn $45/hr stuffing envelopes from home! No experience needed. Must be 18+ and have internet access. Reply YES to this email to apply."),
    ("spam", "Urgent hiring: Online survey takers needed! Earn $3-$75 per survey. Work from home at your own pace. Weekly PayPal payments guaranteed. Register free at survey-cash-now.com"),
    ("spam", "Make $2,000-$5,000 per week from home as a secret shopper! All you need is a smartphone and 2 hours per day. Training provided. First payment in 48 hours. Apply online today."),
    ("spam", "Exciting opportunity: Earn extra income as an Amazon affiliate reseller! No inventory needed. We ship directly to your customers. Earn $500+ weekly. 100% legitimate work-from-home."),

    # ---- Lottery / Prize Scams ----
    ("spam", "CONGRATULATIONS! You have been selected as the winner of our £950,000 jackpot! Your email address was randomly drawn from 2.5 million participants. Reply with your name and address to claim your prize."),
    ("spam", "You are a WINNER! Your email has been selected to receive a $500 Walmart gift card. This is NOT spam - you are a verified winner. Click the link to claim within 24 hours or forfeit your prize."),
    ("spam", "International Email Lottery Results: Your email address has won $750,000 USD in our annual draw. Ticket number: EL-2026-7745XK. Contact our agent at lottery-claim@gmail.com to collect winnings."),
    ("spam", "FINAL NOTICE: Unclaimed prize alert! We have been trying to reach you regarding your $250,000 sweepstakes prize. You have 48 hours to claim before the prize is forfeited. Respond immediately."),

    # ---- Invoice / Business Email Compromise ----
    ("spam", "Urgent invoice notification: Invoice #INV-847203 for $4,850.00 is now 60 days overdue. Failure to pay will result in legal action and additional penalties. Download attached invoice and wire payment immediately."),
    ("spam", "Payment request: Please process the attached wire transfer of $12,500 to our new bank account. Our banking details have changed effective immediately. Please update your records and confirm payment."),
    ("spam", "FINAL DEMAND: Your outstanding balance of $3,200 must be paid within 24 hours to avoid referral to our collections agency and damage to your credit score. Click here to pay now."),

    # ---- Generic / Catch-All Spam ----
    ("spam", "You have been pre-approved for a personal loan of up to $35,000 with no credit check required! Low interest rates starting at 2.9%. Apply in 60 seconds at easy-loans-approved.com"),
    ("spam", "FREE iPhone 15 Pro! Complete a short 3-question survey and claim your brand new iPhone. No purchase necessary. Limited quantities available. Survey ends tonight at midnight!"),
    ("spam", "Lose 30 pounds in 30 days with our revolutionary diet pill! Doctors hate this one weird trick! 100% natural ingredients. Money-back guarantee. Order now and get 50% off!"),
    ("spam", "Your computer has a VIRUS! Microsoft has detected dangerous malware on your PC. Call our toll-free tech support number immediately: 1-800-555-0192. Do not turn off your computer!"),
    ("spam", "Exclusive member offer: As a valued customer, you have been selected for our VIP rewards program. Claim your $200 bonus cash by clicking here and entering code: VIP2026"),

    # ---- Legitimate Ham (non-spam) examples to prevent overfit ----
    ("ham", "Hi, can we reschedule tomorrow's 2pm meeting to 3pm? I have a conflict with another call."),
    ("ham", "Please find attached the Q3 financial report for your review. Let me know if you have any questions."),
    ("ham", "Your order #A28374 has been shipped and will arrive by Friday. Track your package using the link in your account."),
    ("ham", "Reminder: Your dentist appointment is scheduled for Monday December 2nd at 10:30am. Reply YES to confirm."),
    ("ham", "Hi! Are you free for coffee this weekend? Let me know what works for you."),
    ("ham", "The team meeting notes from today are attached. Action items are due by end of next week."),
    ("ham", "Your flight confirmation: AA1234 departing JFK at 8:45am on Nov 20. Gate information available 24 hours before departure."),
    ("ham", "Password reset confirmation: Your password was successfully changed. If you did not make this change, contact support immediately."),
    ("ham", "Monthly statement: Your account balance as of Nov 30 is $2,435.67. No action required."),
    ("ham", "Hi, the project deadline has been extended to December 15. Please update your task tracker accordingly."),
]
