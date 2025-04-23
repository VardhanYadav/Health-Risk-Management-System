var divElement = document.getElementById('viz1741508904979');                    
                        var vizElement = divElement.getElementsByTagName('object')[0];                    
                        if ( divElement.offsetWidth > 800 ) 
                        { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
                        else if ( divElement.offsetWidth > 500 ) 
                        { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
                        else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     
                        var scriptElement = document.createElement('script');                    
                        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
                        vizElement.parentNode.insertBefore(scriptElement, vizElement);                
                        

        // User authentication state
        let isAuthenticated = false;
        const demoUsers = [
            { username: "admin", password: "admin123", name: "Administrator", role: "admin" },
            { username: "doctor", password: "doctor123", name: "Dr. Smith", role: "doctor" },
            { username: "nurse", password: "nurse123", name: "Nurse Johnson", role: "nurse" }
        ];

        // DOM Elements
        const loginModal = document.getElementById('loginModal');
        const loginBtn = document.getElementById('loginBtn');
        const closeModal = document.querySelectorAll('.close-modal');
        const loginForm = document.getElementById('loginForm');
        const registerLink = document.getElementById('registerLink');
        const registrationModal = document.getElementById('registrationModal');
        const registrationForm = document.getElementById('registrationForm');
        const loginFromRegister = document.getElementById('loginFromRegister');
        const scrollIndicator = document.getElementById('scrollTop');
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        const fullscreenRiskCalc = document.getElementById('fullscreenRiskCalc');
        const expandRiskCalc = document.getElementById('expandRiskCalc');
        const closeFullscreen = document.getElementById('closeFullscreen');
        const mainHeader = document.getElementById('mainHeader');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const notificationToast = document.getElementById('notificationToast');
        const toastMessage = document.getElementById('toastMessage');
        const reportModal = document.getElementById('reportModal');
        const reportBody = document.getElementById('reportBody');
        const reportDate = document.getElementById('reportDate');
        const reportPolicyType = document.getElementById('reportPolicyType');
        const reportCoverage = document.getElementById('reportCoverage');
        const reportAge = document.getElementById('reportAge');
        const downloadReportBtn = document.getElementById('downloadReportBtn');
        const printReportBtn = document.getElementById('printReportBtn');
        const emailReportBtn = document.getElementById('emailReportBtn');
        const generateReportBtn = document.getElementById('generateReportBtn');
        const resetSelectionBtn = document.getElementById('resetSelectionBtn');
        const coverageAmount = document.getElementById('coverageAmount');
        const coverageValue = document.getElementById('coverageValue');
        const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
        const comparisonResults = document.getElementById('comparisonResults');
        const resultsGrid = document.getElementById('resultsGrid');

        // Insurance company data
        const insuranceData = {
            lic: {
                name: "LIC",
                logo: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlE3xnOPiPJ3SuoNbYru_UHW1R7WXPI9ojcw&s",
                width :'50',
                height:'50',
                premiumRange: "₹1,000 - ₹10,000",
                coverageLimit: "No specific limit",
                claimRatio: "98.31%",
                rating: 4.5,
                features: ["High claim settlement ratio", "Wide range of policies", "Government-backed", "Long-term investment options"],
                pros: ["Trusted brand", "High maturity benefits", "Flexible premium payment options"],
                cons: ["Higher premiums for some policies", "Limited online services"],
                premiumEstimate: function(coverage, age) {
                    // Simple estimation algorithm
                    const base = 1000;
                    const ageFactor = age.includes("18-25") ? 0.8 : 
                                     age.includes("26-35") ? 1.0 :
                                     age.includes("36-45") ? 1.2 :
                                     age.includes("46-55") ? 1.5 : 2.0;
                    const coverageFactor = coverage / 5000000;
                    return Math.round(base * ageFactor * coverageFactor * 100) / 100;
                }
            },
            hdfc: {
                name: "HDFC Life",
                logo: "https://play-lh.googleusercontent.com/POGVZhNvSh05yA01H2VTlElv0Mw6r4R7hj9w7DaOvJVgiGoi0Fcawi02yITkyMT1zwqO",
                width: '50', 
                height:'50',
                premiumRange: "₹500 - ₹8,000",
                coverageLimit: "Up to ₹20 Cr",
                claimRatio: "99.04%",
                rating: 4.5,
                features: ["Comprehensive term plans", "High coverage options", "Online policy management", "Riders available"],
                pros: ["Competitive premiums", "Quick claim settlement", "Good customer service"],
                cons: ["Medical tests required for higher coverage", "Limited branch network in rural areas"],
                premiumEstimate: function(coverage, age) {
                    const base = 800;
                    const ageFactor = age.includes("18-25") ? 0.7 : 
                                     age.includes("26-35") ? 0.9 :
                                     age.includes("36-45") ? 1.1 :
                                     age.includes("46-55") ? 1.4 : 1.8;
                    const coverageFactor = coverage / 5000000;
                    return Math.round(base * ageFactor * coverageFactor * 100) / 100;
                }
            },
            icici: {
                name: "ICICI Prudential",
                logo: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTi3aJpgJ2xldshLEVzmXv2lX7z2NFFmykjg&s",width: '50', height: '50',
                premiumRange: "₹600 - ₹9,500",
                coverageLimit: "Up to ₹25 Cr",
                claimRatio: "98.74%",
                rating: 4.2,
                features: ["Flexible premium payment terms", "Online policy management", "Multiple riders available", "Wealth creation options"],
                pros: ["Good online services", "Flexible payment options", "High coverage limits"],
                cons: ["Premium rates can be high for older ages", "Complex product portfolio"],
                premiumEstimate: function(coverage, age) {
                    const base = 900;
                    const ageFactor = age.includes("18-25") ? 0.75 : 
                                     age.includes("26-35") ? 0.95 :
                                     age.includes("36-45") ? 1.15 :
                                     age.includes("46-55") ? 1.45 : 1.85;
                    const coverageFactor = coverage / 5000000;
                    return Math.round(base * ageFactor * coverageFactor * 100) / 100;
                }
            },
            sbi: {
                name: "SBI Life",
                logo: "https://yt3.googleusercontent.com/rUWPJAjTrYmNiJjX2D4whywLrdgJUoSxnkk-QglB2zBFgMK2Z2hRGEr-XCrlAYaGxQfh3l1_cg0=s900-c-k-c0x00ffffff-no-rj", width :'50', height:'50',
                premiumRange: "₹700 - ₹7,500",
                coverageLimit: "Up to ₹15 Cr",
                claimRatio: "98.03%",
                rating: 4.1,
                features: ["Affordable term plans", "Simple products", "Bank assurance support", "Quick claim processing"],
                pros: ["Affordable premiums", "Backed by SBI", "Simple product structure"],
                cons: ["Limited online services", "Fewer product options"],
                premiumEstimate: function(coverage, age) {
                    const base = 700;
                    const ageFactor = age.includes("18-25") ? 0.8 : 
                                     age.includes("26-35") ? 1.0 :
                                     age.includes("36-45") ? 1.2 :
                                     age.includes("46-55") ? 1.5 : 1.9;
                    const coverageFactor = coverage / 5000000;
                    return Math.round(base * ageFactor * coverageFactor * 100) / 100;
                }
            },
            max: {
                name: "Max Life",
                logo: "https://bsmedia.business-standard.com/_media/bs/img/article/2021-05/21/full/20210521150122.jpg", width :'50', height: '50',
                premiumRange: "₹800 - ₹12,000",
                coverageLimit: "Up to ₹30 Cr",
                claimRatio: "99.35%",
                rating: 4.6,
                features: ["High claim settlement ratio", "Comprehensive riders", "Online services", "Flexible payment options"],
                pros: ["Highest claim settlement ratio", "Good customer service", "Comprehensive coverage"],
                cons: ["Higher premiums", "Complex product structure"],
                premiumEstimate: function(coverage, age) {
                    const base = 1200;
                    const ageFactor = age.includes("18-25") ? 0.7 : 
                                     age.includes("26-35") ? 0.9 :
                                     age.includes("36-45") ? 1.1 :
                                     age.includes("46-55") ? 1.4 : 1.8;
                    const coverageFactor = coverage / 5000000;
                    return Math.round(base * ageFactor * coverageFactor * 100) / 100;
                }
            }
        };

        // Show loading spinner
        function showLoading() {
            loadingSpinner.style.display = 'flex';
        }
        
        // Hide loading spinner
        function hideLoading() {
            loadingSpinner.style.display = 'none';
        }
        
        // Show notification toast
        function showToast(message, type = 'success', duration = 3000) {
            toastMessage.textContent = message;
            notificationToast.className = `toast ${type}`;
            notificationToast.classList.add('show');
            
            setTimeout(() => {
                notificationToast.classList.remove('show');
            }, duration);
        }
        
        // Format currency with commas
        function formatCurrency(amount) {
            return '₹' + parseInt(amount).toLocaleString('en-IN');
        }
        
        // Initialize with welcome toast
        setTimeout(() => {
            showToast('Welcome to HRMS Portal! Explore our healthcare risk management tools.', 'success', 5000);
        }, 1000);
        
        // Header scroll effect
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                mainHeader.classList.add('scrolled');
            } else {
                mainHeader.classList.remove('scrolled');
            }
            
            // Scroll indicator
            if (window.scrollY > 300) {
                scrollIndicator.classList.add('visible');
            } else {
                scrollIndicator.classList.remove('visible');
            }
        });
        
        // Show login modal
        loginBtn.addEventListener('click', function(e) {
            e.preventDefault();
            loginModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        });

        // Close modals
        closeModal.forEach(btn => {
            btn.addEventListener('click', function() {
                loginModal.style.display = 'none';
                registrationModal.style.display = 'none';
                reportModal.style.display = 'none';
                document.body.style.overflow = 'auto';
            });
        });

        // Close modal when clicking outside
        window.addEventListener('click', function(e) {
            if (e.target === loginModal || e.target === registrationModal || e.target === reportModal) {
                loginModal.style.display = 'none';
                registrationModal.style.display = 'none';
                reportModal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });

        // Login form submission
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            showLoading();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            // Simulate API call delay
            setTimeout(() => {
                // Check credentials
                const user = demoUsers.find(u => u.username === username && u.password === password);
                
                if (user) {
                    isAuthenticated = true;
                    loginBtn.innerHTML = `<i class="fas fa-user"></i> ${user.name}`;
                    loginModal.style.display = 'none';
                    document.body.style.overflow = 'auto';
                    showToast(`Welcome back, ${user.name}!`, 'success');
                    
                    // Update UI based on role
                    updateUIForRole(user.role);
                } else {
                    showToast('Invalid credentials. Please try again.', 'error');
                }
                
                hideLoading();
            }, 1000);
        });
        
        // Update UI based on user role
        function updateUIForRole(role) {
            // In a real app, you would customize the UI based on user role
            console.log(`Updating UI for ${role} role`);
        }

        // Show registration modal
        registerLink.addEventListener('click', function(e) {
            e.preventDefault();
            loginModal.style.display = 'none';
            registrationModal.style.display = 'flex';
        });
        
        // Switch to login from registration
        loginFromRegister.addEventListener('click', function(e) {
            e.preventDefault();
            registrationModal.style.display = 'none';
            loginModal.style.display = 'flex';
        });
        
        // Registration form submission
        registrationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            showLoading();
            
            const name = document.getElementById('reg-name').value;
            const email = document.getElementById('reg-email').value;
            const username = document.getElementById('reg-username').value;
            const password = document.getElementById('reg-password').value;
            const confirm = document.getElementById('reg-confirm').value;
            const role = document.getElementById('reg-role').value;
            
            setTimeout(() => {
               
                if (password !== confirm) {
                    showToast('Passwords do not match!', 'error');
                    hideLoading();
                    return;
                }
                
                if (demoUsers.some(u => u.username === username)) {
                    showToast('Username already exists. Please choose another.', 'error');
                    hideLoading();
                    return;
                }
                
                demoUsers.push({
                    username: username,
                    password: password,
                    name: name,
                    email: email,
                    role: role
                });
                
                showToast('Registration successful! You can now login.', 'success');
                registrationModal.style.display = 'none';
                loginModal.style.display = 'flex';
                document.body.style.overflow = 'auto';
                hideLoading();
                
                registrationForm.reset();
            }, 1500);
        });

        expandRiskCalc.addEventListener('click', function() {
            fullscreenRiskCalc.style.display = 'block';
            document.body.style.overflow = 'hidden';
        });
        
        closeFullscreen.addEventListener('click', function() {
            fullscreenRiskCalc.style.display = 'none';
            document.body.style.overflow = 'auto';
        });

        scrollIndicator.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });

        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanes.forEach(pane => pane.classList.remove('active'));
                
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
                
                document.getElementById(tabId).style.animation = 'fadeIn 0.5s ease-out';
            });
        });

        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {

                if (this.classList.contains('tab-btn')) return;
                
                e.preventDefault();
                
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 100,
                        behavior: 'smooth'
                    });
                    
                    if (targetId === '#insurance' || targetId === '#risk' || targetId === '#policy-comparison') {
                        const tabBtn = document.querySelector(`.tab-btn[data-tab="${targetId.substring(1)}"]`);
                        if (tabBtn) {
                            tabButtons.forEach(btn => btn.classList.remove('active'));
                            tabPanes.forEach(pane => pane.classList.remove('active'));
                            
                            tabBtn.classList.add('active');
                            document.getElementById(targetId.substring(1)).classList.add('active');
                        }
                    }
                }
            });
        });

        function checkAuth() {
            if (!isAuthenticated) {
                loginModal.style.display = 'flex';
                document.body.style.overflow = 'hidden';
                showToast('Please login to access this feature', 'error');
                return false;
            }
            return true;
        }

        document.querySelectorAll('.protected').forEach(element => {
            element.addEventListener('click', function(e) {
                if (!checkAuth()) {
                    e.preventDefault();
                }
            });
        });
        
        document.querySelectorAll('button, a.btn, .header-cta').forEach(button => {
            button.addEventListener('mousedown', function() {
                this.style.transform = 'translateY(1px)';
            });
            
            button.addEventListener('mouseup', function() {
                this.style.transform = '';
            });
            
            button.addEventListener('mouseleave', function() {
                this.style.transform = '';
            });
        });
        
        document.querySelectorAll('.btn, .login-btn, .header-cta').forEach(button => {
            button.addEventListener('click', function(e) {
                const x = e.clientX - e.target.getBoundingClientRect().left;
                const y = e.clientY - e.target.getBoundingClientRect().top;
                
                const ripple = document.createElement('span');
                ripple.classList.add('ripple');
                ripple.style.left = `${x}px`;
                ripple.style.top = `${y}px`;
                
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 1000);
            });
        });
        
        function addPulseAnimation() {
            const elements = document.querySelectorAll('.btn-primary, .mission-card, .feature-card:first-child');
            elements.forEach((el, index) => {
                setTimeout(() => {
                    el.style.animation = 'pulse 2s infinite';
                }, index * 500);
            });
        }
        
        // Update coverage amount display
        coverageAmount.addEventListener('input', function() {
            coverageValue.textContent = formatCurrency(this.value);
        });
        
        // Compare checkboxes functionality
        compareCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const row = this.closest('.comparison-row');
                if (this.checked) {
                    row.classList.add('selected');
                } else {
                    row.classList.remove('selected');
                }
                updateComparisonResults();
            });
        });
        
        // Update comparison results
        function updateComparisonResults() {
            const selectedCompanies = [];
            compareCheckboxes.forEach(checkbox => {
                if (checkbox.checked) {
                    const companyId = checkbox.closest('.comparison-row').getAttribute('data-company');
                    selectedCompanies.push(companyId);
                }
            });
            
            if (selectedCompanies.length >= 2) {
                comparisonResults.style.display = 'block';
                resultsGrid.innerHTML = '';
                
                const policyType = document.getElementById('policyType').value;
                const coverage = parseInt(document.getElementById('coverageAmount').value);
                const ageRange = document.getElementById('ageRange').value;
                
                selectedCompanies.forEach(companyId => {
                    const company = insuranceData[companyId];
                    const premiumEstimate = company.premiumEstimate(coverage, ageRange);
                    
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.innerHTML = `
                        <h5><img src="${company.logo}" alt="${company.name} Logo" class="company-logo"> ${company.name}</h5>
                        <ul>
                            <li><span>Estimated Premium:</span> <span>₹${premiumEstimate.toLocaleString('en-IN')}/month</span></li>
                            <li><span>Coverage Limit:</span> <span>${company.coverageLimit}</span></li>
                            <li><span>Claim Settlement Ratio:</span> <span>${company.claimRatio}</span></li>
                            <li><span>Customer Rating:</span> <span>${company.rating}/5</span></li>
                        </ul>
                        <h6>Key Features:</h6>
                        <ul>
                            ${company.features.map(feature => `<li><span>•</span> <span>${feature}</span></li>`).join('')}
                        </ul>
                        <h6>Pros:</h6>
                        <ul>
                            ${company.pros.map(pro => `<li><span>•</span> <span>${pro}</span></li>`).join('')}
                        </ul>
                        <h6>Cons:</h6>
                        <ul>
                            ${company.cons.map(con => `<li><span>•</span> <span>${con}</span></li>`).join('')}
                        </ul>
                    `;
                    resultsGrid.appendChild(card);
                });
            } else {
                comparisonResults.style.display = 'none';
            }
        }
        
        // Generate report
        generateReportBtn.addEventListener('click', function() {
            const selectedCompanies = [];
            compareCheckboxes.forEach(checkbox => {
                if (checkbox.checked) {
                    const companyId = checkbox.closest('.comparison-row').getAttribute('data-company');
                    selectedCompanies.push(companyId);
                }
            });
            
            if (selectedCompanies.length < 2) {
                showToast('Please select at least two policies to compare', 'error');
                return;
            }
            
            const policyType = document.getElementById('policyType').value;
            const coverage = parseInt(document.getElementById('coverageAmount').value);
            const ageRange = document.getElementById('ageRange').value;
            
            // Set report metadata
            reportDate.textContent = new Date().toLocaleDateString('en-US', {
                year: 'numeric', month: 'long', day: 'numeric'
            });
            reportPolicyType.textContent = document.getElementById('policyType').options[document.getElementById('policyType').selectedIndex].text;
            reportCoverage.textContent = formatCurrency(coverage);
            reportAge.textContent = document.getElementById('ageRange').options[document.getElementById('ageRange').selectedIndex].text;
            
            // Generate report content
            reportBody.innerHTML = `
                <h4>Policy Comparison Summary</h4>
                <p>Comparison of ${selectedCompanies.length} insurance policies based on your selected criteria.</p>
                
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Company</th>
                            <th>Premium Estimate</th>
                            <th>Coverage Limit</th>
                            <th>Claim Ratio</th>
                            <th>Rating</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${selectedCompanies.map(companyId => {
                            const company = insuranceData[companyId];
                            const premiumEstimate = company.premiumEstimate(coverage, ageRange);
                            return `
                                <tr>
                                    <td><img src="${company.logo}" alt="${company.name} Logo" class="company-logo"> ${company.name}</td>
                                    <td>₹${premiumEstimate.toLocaleString('en-IN')}/month</td>
                                    <td>${company.coverageLimit}</td>
                                    <td>${company.claimRatio}</td>
                                    <td>${company.rating}/5</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
                
                <h4>Detailed Comparison</h4>
                ${selectedCompanies.map(companyId => {
                    const company = insuranceData[companyId];
                    const premiumEstimate = company.premiumEstimate(coverage, ageRange);
                    return `
                        <div style="margin-bottom: 30px;">
                            <h5><img src="${company.logo}" alt="${company.name} Logo" class="company-logo"> ${company.name}</h5>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                                <div>
                                    <h6>Key Features</h6>
                                    <ul>
                                        ${company.features.map(feature => `<li> ${feature}</li>`).join('')}
                                    </ul>
                                </div>
                                <div>
                                    <h6>Pros & Cons</h6>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                        <div>
                                            <h6>Pros</h6>
                                            <ul>
                                                ${company.pros.map(pro => `<li> ${pro}</li>`).join('')}
                                            </ul>
                                        </div>
                                        <div>
                                            <h6>Cons</h6>
                                            <ul>
                                                ${company.cons.map(con => `<li> ${con}</li>`).join('')}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
                
                <div class="recommendation-card">
                    <h5>Recommendation</h5>
                    <p>Based on your selected criteria (${document.getElementById('ageRange').options[document.getElementById('ageRange').selectedIndex].text}, ${formatCurrency(coverage)} coverage), we recommend considering the following:</p>
                    <ul>
                        <li>For lowest premium: <strong>${selectedCompanies.reduce((min, companyId) => {
                            const current = insuranceData[companyId].premiumEstimate(coverage, ageRange);
                            return current < min.premium ? {company: companyId, premium: current} : min;
                        }, {company: null, premium: Infinity}).company}</strong></li>
                        <li>For highest claim settlement ratio: <strong>${selectedCompanies.reduce((max, companyId) => {
                            const current = parseFloat(insuranceData[companyId].claimRatio);
                            return current > max.ratio ? {company: companyId, ratio: current} : max;
                        }, {company: null, ratio: 0}).company}</strong></li>
                        <li>For best customer rating: <strong>${selectedCompanies.reduce((max, companyId) => {
                            const current = insuranceData[companyId].rating;
                            return current > max.rating ? {company: companyId, rating: current} : max;
                        }, {company: null, rating: 0}).company}</strong></li>
                    </ul>
                </div>
            `;
            
            reportModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        });
        
        // Reset selection
        resetSelectionBtn.addEventListener('click', function() {
            compareCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
                checkbox.closest('.comparison-row').classList.remove('selected');
            });
            comparisonResults.style.display = 'none';
            showToast('Selection cleared', 'success');
        });
        
        // Download report as PDF (mock functionality)
        downloadReportBtn.addEventListener('click', function() {
            showLoading();
            setTimeout(() => {
                hideLoading();
                showToast('Report download started (mock functionality)', 'success');
                
                // In a real implementation, this would generate and download a PDF
                // For demo purposes, we'll just show a toast
            }, 1500);
        });
        
        // Print report
        printReportBtn.addEventListener('click', function() {
            window.print();
        });
        
        // Email report (mock functionality)
        emailReportBtn.addEventListener('click', function() {
            showLoading();
            setTimeout(() => {
                hideLoading();
                showToast('Report sent to your email (mock functionality)', 'success');
            }, 1500);
        });
        
        window.addEventListener('load', function() {
            addPulseAnimation();
            
            const animateOnScroll = function() {
                const elements = document.querySelectorAll('.feature-card, .service-card, .step-card');
                
                elements.forEach(element => {
                    const elementPosition = element.getBoundingClientRect().top;
                    const windowHeight = window.innerHeight;
                    
                    if (elementPosition < windowHeight - 100) {
                        element.style.opacity = '1';
                        element.style.transform = 'translateY(0)';
                    }
                });
            };
            
            // Set initial state
            document.querySelectorAll('.feature-card, .service-card, .step-card').forEach(element => {
                element.style.opacity = '0';
                element.style.transform = 'translateY(20px)';
                element.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
            });
            
            window.addEventListener('scroll', animateOnScroll);
            animateOnScroll(); // Run once on load
        });